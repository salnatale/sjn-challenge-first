from typing import Iterable
from collections import defaultdict
from typing import Optional, Callable

BONUS_INCREMENT = 10  # bonus offered in $10 increments
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

    def all_ancestors(self, user: str) -> list[str]:
        """for use in later steps
        Walk up through parents to find all ancestors (direct or indirect)."""
        result = []
        current = user
        while current in self._parents:
            current = self._parents[current]
            result.append(current)
        return result

    def get_all_nodes(self) -> set[str]:
        """Return all nodes in the network."""
        return self._nodes.copy() # so as not to work w/ funky mutable objects.


# =============================================================================
# Part 2: Influence Metrics (pure functions - do not mutate network)
# =============================================================================

def top_k_by_reach(network: ReferralNetwork, k: int) -> list[str]:
    """
    Rank users by Reach: number of distinct descendants.
    Returns top k users sorted by reach descending.
    """
    reach = {user: len(list(network.all_referrals(user))) for user in network.get_all_nodes()}
    return sorted(reach, key=lambda u: reach[u], reverse=True)[:k]


def top_k_by_flow_centrality(network: ReferralNetwork, k: int) -> list[str]:
    """
    Rank users by Flow Centrality.
    defined as: 
    number of ordered pairs (s, t) for which u lies on the shortest directed path from s to t,
    where s and t are distinct users and u is a user between them.
    """
    # naive, iterate over all pairs of users and count the number of times u lies on the shortest directed path from s to t.
    # we can avoid this by checking: 
    #   - s is an ancestor of u, AND
    #   - t is a descendant of u
    # Because u is on path s→t iff above: 
    # also guarantee no ancestors will be in children, so no need to check for that.
    flow = {
        user: len(network.all_ancestors(user)) * len(list(network.all_referrals(user)))
        for user in network.get_all_nodes()
    }
    return sorted(flow, key=lambda u: flow[u], reverse=True)[:k]

# =============================================================================
# Part 3: Growth
# =============================================================================

def expected_network_size(p: float, days: int) -> float:
    """
    Rules:
    - 100 initial referrers, each with capacity 10
    - Each day, active referrer succeeds with prob p (max 1/day)
    - Success consumes 1 capacity; at 0, referrer becomes inactive
    - New referrer joins next day with capacity 10

    expected network size at end of day [days]. initial 100 + num successful referrals. 
    lifetime capacity 10 successful referrals, per user. Any day, active referrer at most 1 successful. 
    """
    def _rebuild_capacity(capacity: list[float], day_successes: float) -> list[float]:
        new_capacity = [0.0] * 11
        # compute changes in expectation.
        for c in range(1, 11):
            # failed: stay at c
            new_capacity[c] += capacity[c] * (1 - p)
            # succeeded: move to c-1
            new_capacity[c - 1] += capacity[c] * p
        # new referrers from today's successes join at capacity 10
        new_capacity[10] += day_successes
        return new_capacity

    # capacity[c] = expected count (float) of referrers with capacity c
    # index 0 = inactive, 1-10 = active
    capacity = [0.0] * 11  # none in each bucket at init. 
    capacity[10] = 100.0  # start with 100 referrers at capacity 10

    total_successes = 0.0

    for _ in range(days + 1):  # days 0 through days (inclusive)
        active = sum(capacity[1:11])
        day_successes = active * p
        total_successes += day_successes

        capacity = _rebuild_capacity(capacity, day_successes)

    return 100.0 + total_successes



# sanity check :
# p=0, days=10: 100.0
# p=1, days=0: 200.0 ## EOD Indexed.
# p=0.5, days=5: 1139.06
# p=0.1, days=20: 740.02


# =============================================================================
# Part 4: Incentive Optimization
# =============================================================================



def min_bonus_for_target(
    days: int,
    target_network_size: int,
    adoption_prob: Callable[[int], float], 
    initial_high: int = BONUS_INCREMENT  # optional initial high bound, perhaps of prior runs. 
) -> Optional[int]:
    """
    Find minimum bonus ($10 increments) to reach target network size.

    Args:
        days: number of days to simulate
        target_network_size: target network size to reach
        adoption_prob: black-box function mapping bonus -> probability (monotonic non-decreasing, expensive)

    Returns:
        Smallest bonus achieving target, or None if impossible.
    """
    def reaches_target(bonus: int) -> bool:
        p = adoption_prob(bonus)
        return expected_network_size(p, days) >= target_network_size

    # Strategy: binary search over expensive function, to reduce overall cost to O(log n) calls to adoption_prob.
    
    # Phase 1: find upper bound, exponentially increasing + maybe we have reference we can inject? 
    low, high = 0, initial_high
    while not reaches_target(high):
        if adoption_prob(high) >= 1.0:
            return None  # max probability, still can't reach
        low = high
        high *= 2

    # Phase 2: binary search for exact correct bonus amount. 
    while low + BONUS_INCREMENT <= high:
        mid = round((low + high) / 2 / BONUS_INCREMENT) * BONUS_INCREMENT  # instead of default round to support all increments.
        if reaches_target(mid):
            high = mid
        else:
            low = mid + BONUS_INCREMENT

    return low