from typing import Iterable
from collections import defaultdict


class ReferralError(ValueError):
    pass


class ReferralNetwork:
    """
    A directed graph where edges represent referrer → candidate relationships.

    Invariants:
    - No self-referrals
    - Each candidate has at most one referrer
    - Acyclic (no cycles allowed)
    """

    def __init__(self):
        # optimized submodels for o(1) or o(chilren) lookup
        self._parents: dict[str, str] = {}  # key candidate : value referrer o(1) instead of o(n) for every add
        self._children: defaultdict[str, set[str]] = defaultdict(set)  # key: referrer: value candidates, without, would have to iterate over parent dict to find direct referrals.
        self._nodes: set[str] = set()
    
    def _check_constraints(self, referrer: str, candidate: str) -> None:
        """
        Check if adding edge referrer to candidate satisfies invariants.
        Raises ReferralError if invalid.
        """
        if referrer == candidate:
            raise ReferralError("Self-referral is not allowed.")

        if candidate in self._parents:
            raise ReferralError(f"{candidate} already has a referrer.")

        # traverse from referrer through parents subdict to check if candidate is an ancestor
        # If candidate is ancestor of referrer, adding referrer → candidate creates cycle
        current = referrer
        while current in self._parents:
            current = self._parents[current]
            if current == candidate:
                raise ReferralError("Adding this referral would create a cycle.")




    def add_referral(self, referrer: str, candidate: str) -> None:
        """Add edge referrer → candidate. Raises ReferralError if constraints violated."""
        self._check_constraints(referrer, candidate)
        # a- OK, add.
        self._parents[candidate] = referrer
        self._children[referrer].add(candidate)
        # brute try add
        self._nodes.add(referrer)
        self._nodes.add(candidate)

    def direct_referrals(self, user: str) -> Iterable[str]:
        """Return immediate children of user, if the user exists"""
        return self._children.get(user, set())


    def all_referrals(self, user: str) -> Iterable[str]:
        """DFS through graph to find all descendents either direct or indirect."""
        result = []
        stack = list(self._children.get(user, []))
        while stack:
            node = stack.pop()
            result.append(node)
            stack.extend(self._children.get(node, []))
        return result

    def get_all_nodes(self) -> set[str]:
        """Return all nodes in the network."""
        return self._nodes.copy() # so as not to work w/ funky mutable objects. 