import re

import leantree.utils
from leantree.repl_adapter.data import SingletonProofTree, SingletonProofTreeNode, SingletonProofTreeEdge
from leantree.file_span import FileSpan
from leantree.repl_adapter.ast_parser import LeanASTObject, LeanASTArray


class ProofTreePostprocessor:
    @classmethod
    def transform_proof_tree(cls, tree: SingletonProofTree):
        def visitor(node: SingletonProofTreeNode):
            assert node.tactic is not None, "Transforming an unsolved node."
            if node.tactic.is_synthetic():
                return

            # Note: the order is important here, because tacticStrings are being modified.
            cls._replace_nested_tactics_with_sorries(node)
            # _remove_by_sorry_in_have is disabled: in Lean 4.27+, bodyless `have h : T`
            # (without `:= by sorry`) is no longer valid tactic syntax.  Keeping the
            # `:= by sorry` body works correctly — the REPL creates a sorry branch that
            # the tree builder matches to the spawned_goals child.
            cls._transform_with_cases(node)
            cls._transform_case_tactic(node)
            cls._transform_simp_rw(node)
            cls._transform_rw(node)
            cls._transform_calc(node)

            node.tactic.tactic_string = leantree.utils.remove_empty_lines(leantree.utils.remove_comments(
                node.tactic.tactic_string
            ))

        cls._add_missing_assumption_tactics(tree)
        tree.traverse_preorder(visitor)

    @classmethod
    def _add_missing_assumption_tactics(cls, tree: SingletonProofTree):
        if tree.is_solved():
            return

        # e.g. `by` or `suffices` tactics seem to transform the spawned goal to a state where the goal trivially follows
        # from a hypothesis, but then no `assumption` tactic follows. Unfortunately it is not enough to compare the goal
        # type with all hypotheses syntactically - consider "x ≠ 1" and "¬x = 1".
        # This fix seems reckless, but the resulting trees are later verified for correctness.
        def visitor(node: SingletonProofTreeNode):
            if node.tactic is None:
                node.set_edge(SingletonProofTreeEdge.create_synthetic(
                    tactic_string="assumption",
                    goal_before=node.goal,
                    goals_after=[],
                    spawned_goals=[],
                ))

        tree.traverse_preorder(visitor)

    # https://lean-lang.org/doc/reference/latest//Tactic-Proofs/Tactic-Reference/#cases
    @classmethod
    def _transform_with_cases(cls, node: SingletonProofTreeNode):
        tactic_str = node.tactic.tactic_string
        cases_match = re.search(r"^(cases\s+[^\n]+)with\s+", tactic_str)
        induction_match = re.search(r"^(induction\s+[^\n]+)with\s+", tactic_str)
        if cases_match:
            if node.tactic.ast is None:
                return
            constructors = cls._extract_cases_constructors(node)
            match = cases_match
        elif induction_match:
            if node.tactic.ast is None:
                return
            constructors = cls._extract_induction_constructors(node)
            match = induction_match
        else:
            return

        if constructors is None or len(constructors) != len(node.tactic.spawned_goals):
            # Either inductionAlts not found, or BetterParser already decomposed the
            # alternatives into separate steps (so spawned_goals is empty/mismatched).
            # Just trim the "with ..." suffix from the tactic string.
            node.tactic.tactic_string = match.group(1).rstrip()
            return
        intermezzo_nodes = []
        for constructor, child in zip(constructors, node.tactic.spawned_goals):
            # We do not want to synthesize the state before the renaming of constructor variables, so we leave that to
            # Lean during tree verification.
            intermezzo_node = SingletonProofTreeNode.create_synthetic(
                parent=node,
            )
            # The `case` tactic handles renaming of inaccessible hypotheses.
            intermezzo_node.set_edge(SingletonProofTreeEdge.create_synthetic(
                tactic_string=f"case {constructor}",
                goal_before=intermezzo_node.goal,
                spawned_goals=[child],
                goals_after=[],
            ))
            intermezzo_nodes.append(intermezzo_node)
        node.tactic.spawned_goals = intermezzo_nodes
        node.tactic.tactic_string = match.group(1)

        # Alternatively, we could use the explicit `rename_i` tactic in each branch to not depend on Mathlib.
        # https://lean-lang.org/doc/reference/latest/Tactic-Proofs/The-Tactic-Language/#rename_i

        # Another idea would be to use the cases' tactic from Mathlib.
        # https://leanprover-community.github.io/mathlib4_docs/Mathlib/Tactic/Cases.html#Mathlib.Tactic.cases'
        # However, cases' doesn't work because not all argument names in the constructors of "cases ... with" need to be specified,
        # so the constructor arguments names and the cases' arguments would be misaligned. We could align them by using "_"
        # in `cases'`, but for that we would need to know the number of arguments for each constructor (which is not visible
        # from the AST)

    # Note that `case` tactics are still present in the tree because they handle variable renaming (not just goal selection).
    @classmethod
    def _transform_case_tactic(cls, node: SingletonProofTreeNode):
        # https://lean-lang.org/doc/reference/latest//Tactic-Proofs/The-Tactic-Language/#case
        tactic_str = node.tactic.tactic_string
        pattern = r"case'?[ \t]+([^\n]+?)[ \t]+=>"
        match = re.match(pattern, tactic_str)
        if not match:
            return

        # Note `case'` doesn't force the goal to be solved immediately, but `case` seems to work as well in the REPL.
        new_tactic = f"case {match.group(1)}"
        node.tactic.tactic_string = new_tactic

    # TODO: e.g. `have` doesn't need the `:= by sorry` - without it, it correctly spawns a goal
    # TODO: expand the spans by any whitespaces at the sides
    @classmethod
    def _replace_nested_tactics_with_sorries(cls, node: SingletonProofTreeNode):
        ancestors = [n for n in node.get_subtree_nodes() if n != node]
        # By blocks are present e.g. in
        # https://lean-lang.org/doc/reference/latest//Tactic-Proofs/The-Tactic-Language/#have
        # and replacing the is in accordance with the examples in the official repo. See e.g.:
        # https://github.com/leanprover-community/repl/blob/master/test/name_generator.in
        # By blocks are also in any number of other places, like `exact sum_congr rfl fun x _ ↦ by ac_rfl`.
        sub_spans = []
        for ancestor in ancestors:
            if not ancestor.tactic.is_synthetic() and node.tactic.span.contains(ancestor.tactic.span):
                sub_spans.append(ancestor.tactic.span.relative_to(node.tactic.span.start))
        if sub_spans:
            sub_spans = FileSpan.merge_contiguous_spans(
                sub_spans,
                node.tactic.tactic_string,
                lambda inbetween: len(inbetween.strip()) == 0,
            )
            new_tactic = FileSpan.replace_spans(
                base_string=node.tactic.tactic_string,
                replacement="sorry",
                spans=sub_spans,
            )
            # In Lean 4.27+, `simpa` rejects bare `sorry` in its arguments but
            # accepts `by sorry`.  Apply a targeted fix only for simpa tactics.
            if new_tactic.lstrip().startswith("simpa"):
                new_tactic = re.sub(r'(?<!\bby )(?<!\bby\n)sorry', 'by sorry', new_tactic)
                new_tactic = re.sub(r'by\s+by sorry', 'by sorry', new_tactic)
            node.tactic.tactic_string = new_tactic

    @classmethod
    def _remove_by_sorry_in_have(cls, node: SingletonProofTreeNode):
        match = re.match(r"(have[ \t]+[^\n]+?)[ \t]+:=[ \t]+by[ \t\n]+sorry", node.tactic.tactic_string)
        if not match:
            return
        if len(node.tactic.spawned_goals) != 1:
            return

        node.tactic.tactic_string = match.group(1)
        node.tactic.goals_after.insert(0, node.tactic.spawned_goals[0])
        node.tactic.spawned_goals = []

    @classmethod
    def _find_induction_alts(cls, ast_node, tactic_name: str) -> list[str] | None:
        """Extract constructor names from cases/induction AST by finding the inductionAlts node.
        Returns None if not found — this happens when BetterParser already decomposed the
        alternatives into separate proof steps (each branch is its own step with goal matching)."""
        alts_node = ast_node.find_first_node(
            lambda n: isinstance(n, LeanASTObject) and n.type == "Tactic.inductionAlts"
        )
        if alts_node is None:
            return None
        # Structure can vary — return None if unexpected so caller can skip transform.
        if len(alts_node.args) < 3 or alts_node.args[0].pretty_print() != "with":
            return None
        alts = alts_node.args[2]
        if not isinstance(alts, LeanASTArray):
            return None

        constructors = []
        for alt in alts.items:
            if not (isinstance(alt, LeanASTObject) and alt.type == "Tactic.inductionAlt"):
                return None
            # The expected structure is [LHS, "=>", body]; some variants have only [LHS, body].
            if len(alt.args) < 1:
                return None
            lhs = alt.args[0]
            constructor_tokens = lhs.get_tokens()
            if not constructor_tokens or constructor_tokens[0] != "|":
                return None
            constructor = " ".join(constructor_tokens[1:])
            constructors.append(constructor)
        return constructors

    @classmethod
    def _extract_cases_constructors(cls, node: SingletonProofTreeNode) -> list[str]:
        return cls._find_induction_alts(node.tactic.ast.root, "cases")

    @classmethod
    def _extract_induction_constructors(cls, node: SingletonProofTreeNode) -> list[str]:
        return cls._find_induction_alts(node.tactic.ast.root, "induction")

    @classmethod
    def _transform_simp_rw(cls, node: SingletonProofTreeNode):
        match = re.match(r"simp_rw \[([^\n]+)]( at [^\n]+)?", node.tactic.tactic_string)
        if not match:
            return
        assert len(node.tactic.spawned_goals) == 0, "`simp_rw` has spawned goals"

        rules_list = match.group(1)
        at_clause = match.group(2) or ""

        def simp_only(rule: str) -> str:
            return f"simp only [{rule}]{at_clause}"

        rules = [rule.strip() for rule in rules_list.split(",")]
        assert len(rules) > 0, "No rules in a `simp_rw`"
        if len(rules) == 1:
            return

        node.tactic.tactic_string = simp_only(rules[0])
        goals_after = node.tactic.goals_after
        curr_node = node
        for rule in rules[1:]:
            child = SingletonProofTreeNode.create_synthetic(
                parent=curr_node,
            )
            child.set_edge(SingletonProofTreeEdge.create_synthetic(
                tactic_string=simp_only(rule),
                goal_before=child.goal,
                spawned_goals=[],
                goals_after=[],  # Will be filled in.
            ))
            curr_node.tactic.goals_after = [child]
            child.parent = curr_node

            curr_node = child
        curr_node.tactic.goals_after = goals_after
        for g in goals_after:
            g.parent = curr_node

    # @classmethod
    # def _transform_exacts(cls, node: SingletonProofTreeNode):
    #     match = re.match(r"exacts \[([^\n]+)]", node.tactic.tactic_string.strip())
    #     if not match:
    #         return
    #     print(node.goal)
    #     print()
    #     print(f"tactic: {node.tactic.tactic_string}")
    #     print()
    #     for g in node.parent.tactic.spawned_goals:
    #         print(f"parent spawned: {g.goal}")
    #         print()
    #     for g in node.parent.tactic.goals_after:
    #         print(f"parent after: {g.goal}")
    #         print()
    #     print("------")
    #
    #     terms = match.group(1).split(",")
    #     assert len(node.parent.tactic.all_children()) == len(terms),\
    #         "`exacts` has different number of terms then open goals"
    #     term_idx = [
    #         i for i, child in enumerate(node.parent.tactic.all_children())
    #         if child.goal.semantic_equals(node.goal, ignore_metavars=True)
    #     ]
    #     assert len(term_idx) == 1, "Ambiguous or duplicated open goals for `exacts`"
    #
    #     node.tactic.tactic_string = f"exact {terms[term_idx[0]].strip()}"

    @classmethod
    def _transform_rw(cls, node: SingletonProofTreeNode):
        if node.tactic.tactic_string.strip() == "rw [rfl]":
            node.tactic.tactic_string = "rfl"

    # Decompose `calc` into per-line `have` steps + a final `Trans.trans` combiner.
    # Original:
    #   calc a R1 b := p1
    #        _ R2 c := p2
    #        _ R3 d := p3
    # becomes (N+1 steps, main-line chain):
    #   have h_calc_1 : a R1 b := p1
    #   have h_calc_2 : b R2 c := p2
    #   have h_calc_3 : c R3 d := p3
    #   exact Trans.trans h_calc_1 (Trans.trans h_calc_2 h_calc_3)
    # By-block lines (`:= by tac`) become `have ... := by sorry` with the original by-block's
    # subtree attached as a spawned goal on that have step.
    @classmethod
    def _transform_calc(cls, node: SingletonProofTreeNode):
        if not node.tactic.tactic_string.lstrip().startswith("calc"):
            return
        if node.tactic.ast is None:
            return

        calc_tactic = node.tactic.ast.find_first_node(
            lambda n: isinstance(n, LeanASTObject) and n.type == "Lean.calcTactic"
        )
        if calc_tactic is None:
            return

        steps = cls._parse_calc_steps(calc_tactic)
        if steps is None or len(steps) == 0:
            return

        by_block_count = sum(1 for s in steps if s["is_by"])
        if by_block_count != len(node.tactic.spawned_goals):
            # Unexpected mismatch — bail and leave calc as a single atomic step.
            return

        # Build relation strings per step, substituting `_` with the previous step's RHS.
        relations = []
        prev_rhs = None
        for i, s in enumerate(steps):
            lhs_str = " ".join(s["lhs_tokens"]) if i == 0 else prev_rhs
            rhs_str = " ".join(s["rhs_tokens"])
            relations.append(f"{lhs_str} {s['op']} {rhs_str}")
            prev_rhs = rhs_str

        n = len(steps)

        # Degenerate single-step calc: just `exact <proof>`.
        if n == 1:
            s0 = steps[0]
            node.tactic.tactic_string = f"exact {s0['proof_str']}"
            return

        names = [f"h_calc_{i+1}" for i in range(n)]

        def have_tac(i: int) -> str:
            s = steps[i]
            rhs = "by sorry" if s["is_by"] else s["proof_str"]
            return f"have {names[i]} : {relations[i]} := {rhs}"

        # Right-associative nested Trans.trans combiner.
        combiner_expr = names[-1]
        for name in reversed(names[:-1]):
            if combiner_expr == names[-1]:
                combiner_expr = f"Trans.trans {name} {combiner_expr}"
            else:
                combiner_expr = f"Trans.trans {name} ({combiner_expr})"
        combiner_tac = f"exact {combiner_expr}"

        spawned_iter = iter(node.tactic.spawned_goals)

        def step_spawned(i: int) -> list:
            return [next(spawned_iter)] if steps[i]["is_by"] else []

        original_goals_after = node.tactic.goals_after

        # Rewrite `node` in place as the first have step.
        node.tactic.tactic_string = have_tac(0)
        node.tactic.spawned_goals = step_spawned(0)

        curr = node
        for i in range(1, n):
            next_node = SingletonProofTreeNode.create_synthetic(parent=curr)
            curr.tactic.goals_after = [next_node]
            next_node.parent = curr
            next_node.set_edge(SingletonProofTreeEdge.create_synthetic(
                tactic_string=have_tac(i),
                goal_before=None,
                spawned_goals=step_spawned(i),
                goals_after=[],
            ))
            curr = next_node

        combiner_node = SingletonProofTreeNode.create_synthetic(parent=curr)
        curr.tactic.goals_after = [combiner_node]
        combiner_node.parent = curr
        combiner_node.set_edge(SingletonProofTreeEdge.create_synthetic(
            tactic_string=combiner_tac,
            goal_before=None,
            spawned_goals=[],
            goals_after=original_goals_after,
        ))
        for g in original_goals_after:
            g.parent = combiner_node

    @classmethod
    def _parse_calc_steps(cls, calc_tactic: LeanASTObject) -> list[dict] | None:
        """Extract per-line info from a Lean.calcTactic AST node. Returns None on unexpected shape."""
        steps_node = calc_tactic.find_first_node(
            lambda n: isinstance(n, LeanASTObject) and n.type == "Lean.calcSteps"
        )
        if steps_node is None or len(steps_node.args) < 2:
            return None

        first = steps_node.args[0]
        rest = steps_node.args[1]
        if not (isinstance(first, LeanASTObject) and first.type == "Lean.calcFirstStep"):
            return None
        if not isinstance(rest, LeanASTArray):
            return None

        results = []

        # calcFirstStep args = [relation, [":=" literal, proof]]
        parsed = cls._parse_calc_step_parts(first, is_first=True)
        if parsed is None:
            return None
        results.append(parsed)

        for step in rest.items:
            if not (isinstance(step, LeanASTObject) and step.type == "Lean.calcStep"):
                return None
            parsed = cls._parse_calc_step_parts(step, is_first=False)
            if parsed is None:
                return None
            results.append(parsed)

        return results

    @classmethod
    def _parse_calc_step_parts(cls, step: LeanASTObject, is_first: bool) -> dict | None:
        """Extract relation LHS/op/RHS + proof text + by-block flag from a calc step AST node."""
        if len(step.args) < 2:
            return None
        relation = step.args[0]
        if is_first:
            # args[1] is an array [":=" literal, proof_node]
            proof_container = step.args[1]
            if not isinstance(proof_container, LeanASTArray) or len(proof_container.items) < 2:
                return None
            proof = proof_container.items[-1]
        else:
            # args = [relation, ":=" literal, proof]
            if len(step.args) < 3:
                return None
            proof = step.args[2]

        if not isinstance(relation, LeanASTObject) or len(relation.args) != 3:
            return None
        lhs_node, op_node, rhs_node = relation.args
        if not hasattr(op_node, "value"):
            return None
        op_str = op_node.pretty_print() if hasattr(op_node, "pretty_print") else op_node.value

        lhs_tokens = lhs_node.get_tokens()
        rhs_tokens = rhs_node.get_tokens()
        if not lhs_tokens or not rhs_tokens:
            return None

        is_by = isinstance(proof, LeanASTObject) and proof.type == "Term.byTactic"
        proof_str = " ".join(proof.get_tokens())

        return {
            "lhs_tokens": lhs_tokens,
            "op": op_str,
            "rhs_tokens": rhs_tokens,
            "proof_str": proof_str,
            "is_by": is_by,
        }
