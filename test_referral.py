import pytest
from main import ReferralNetwork, ReferralError, top_k_by_reach, top_k_by_flow_centrality


class TestAddReferral:
    """Tests for add_referral constraints and basic functionality."""

    def test_basic_add(self):
        network = ReferralNetwork()
        network.add_referral("A", "B")
        assert "B" in network.direct_referrals("A")

    def test_chain(self):
        network = ReferralNetwork()
        network.add_referral("A", "B")
        network.add_referral("B", "C")
        network.add_referral("C", "D")
        assert "B" in network.direct_referrals("A")
        assert "C" in network.direct_referrals("B")
        assert "D" in network.direct_referrals("C")

    def test_multiple_children(self):
        network = ReferralNetwork()
        network.add_referral("A", "B")
        network.add_referral("A", "C")
        network.add_referral("A", "D")
        children = set(network.direct_referrals("A"))
        assert children == {"B", "C", "D"}

    def test_self_referral_raises(self):
        network = ReferralNetwork()
        with pytest.raises(ReferralError):
            network.add_referral("A", "A")

    def test_duplicate_referrer_raises(self):
        network = ReferralNetwork()
        network.add_referral("A", "B")
        with pytest.raises(ReferralError):
            network.add_referral("C", "B")  # B already has referrer A

    def test_simple_cycle_raises(self):
        network = ReferralNetwork()
        network.add_referral("A", "B")
        with pytest.raises(ReferralError):
            network.add_referral("B", "A")  # Would create A → B → A

    def test_long_cycle_raises(self):
        network = ReferralNetwork()
        network.add_referral("A", "B")
        network.add_referral("B", "C")
        network.add_referral("C", "D")
        with pytest.raises(ReferralError):
            network.add_referral("D", "A")  # Would create A → B → C → D → A

    def test_atomicity_on_failure(self):
        """Graph should be unchanged after a failed add."""
        network = ReferralNetwork()
        network.add_referral("A", "B")

        try:
            network.add_referral("B", "A")  # Should fail
        except ReferralError:
            pass

        # Graph should still be in original state
        assert "B" in network.direct_referrals("A")
        assert list(network.direct_referrals("B")) == []


class TestDirectReferrals:
    """Tests for direct_referrals method."""

    def test_empty_network(self):
        network = ReferralNetwork()
        assert list(network.direct_referrals("A")) == []

    def test_no_children(self):
        network = ReferralNetwork()
        network.add_referral("A", "B")
        assert list(network.direct_referrals("B")) == []

    def test_multiple_children(self):
        network = ReferralNetwork()
        network.add_referral("A", "B")
        network.add_referral("A", "C")
        assert set(network.direct_referrals("A")) == {"B", "C"}

    def test_only_direct_not_indirect(self):
        network = ReferralNetwork()
        network.add_referral("A", "B")
        network.add_referral("B", "C")
        # A's direct referrals should only be B, not C
        assert set(network.direct_referrals("A")) == {"B"}


class TestAllReferrals:
    """Tests for all_referrals method."""

    def test_empty_network(self):
        network = ReferralNetwork()
        assert list(network.all_referrals("A")) == []

    def test_direct_only(self):
        network = ReferralNetwork()
        network.add_referral("A", "B")
        assert set(network.all_referrals("A")) == {"B"}

    def test_includes_indirect(self):
        network = ReferralNetwork()
        network.add_referral("A", "B")
        network.add_referral("B", "C")
        network.add_referral("C", "D")
        assert set(network.all_referrals("A")) == {"B", "C", "D"}

    def test_tree_structure(self):
        """
        Test tree:
              A
             /|\
            B C D
           /|
          E F
        """
        network = ReferralNetwork()
        network.add_referral("A", "B")
        network.add_referral("A", "C")
        network.add_referral("A", "D")
        network.add_referral("B", "E")
        network.add_referral("B", "F")

        assert set(network.all_referrals("A")) == {"B", "C", "D", "E", "F"}
        assert set(network.all_referrals("B")) == {"E", "F"}
        assert set(network.all_referrals("C")) == set()

    def test_leaf_node(self):
        network = ReferralNetwork()
        network.add_referral("A", "B")
        network.add_referral("B", "C")
        assert list(network.all_referrals("C")) == []


class TestForest:
    """Tests for disconnected components (forest structure)."""

    def test_separate_trees(self):
        network = ReferralNetwork()
        # Tree 1
        network.add_referral("A", "B")
        network.add_referral("B", "C")
        # Tree 2
        network.add_referral("X", "Y")
        network.add_referral("Y", "Z")

        assert set(network.all_referrals("A")) == {"B", "C"}
        assert set(network.all_referrals("X")) == {"Y", "Z"}
        # No cross-tree references
        assert "Y" not in network.all_referrals("A")

    def test_can_add_cross_tree_edge(self):
        """Can connect two separate trees (no cycle)."""
        network = ReferralNetwork()
        network.add_referral("A", "B")
        network.add_referral("X", "Y")
        # Connect tree 2 under tree 1
        network.add_referral("B", "X")

        assert set(network.all_referrals("A")) == {"B", "X", "Y"}


# =============================================================================
# Part 2: Influence Metrics Tests
# =============================================================================

class TestTopKByReach:
    """Tests for top_k_by_reach function."""

    def test_simple_chain(self):
        """A → B → C → D: A has reach 3, B has 2, C has 1, D has 0."""
        network = ReferralNetwork()
        network.add_referral("A", "B")
        network.add_referral("B", "C")
        network.add_referral("C", "D")

        result = top_k_by_reach(network, 4)
        assert result[0] == "A"  # reach = 3
        assert result[1] == "B"  # reach = 2
        assert result[2] == "C"  # reach = 1
        assert result[3] == "D"  # reach = 0

    def test_tree_structure(self):
        """
              A (reach=5)
             /|\
            B C D
           /|
          E F
        B has reach=2, C and D have reach=0
        """
        network = ReferralNetwork()
        network.add_referral("A", "B")
        network.add_referral("A", "C")
        network.add_referral("A", "D")
        network.add_referral("B", "E")
        network.add_referral("B", "F")

        result = top_k_by_reach(network, 2)
        assert result[0] == "A"  # reach = 5
        assert result[1] == "B"  # reach = 2

    def test_k_larger_than_nodes(self):
        """Should return all nodes if k > number of nodes."""
        network = ReferralNetwork()
        network.add_referral("A", "B")

        result = top_k_by_reach(network, 10)
        assert len(result) == 2

    def test_empty_network(self):
        network = ReferralNetwork()
        result = top_k_by_reach(network, 5)
        assert result == []


class TestTopKByFlowCentrality:
    """Tests for top_k_by_flow_centrality function."""

    def test_simple_chain(self):
        """
        A → B → C → D
        B: ancestors=1, descendants=2 → flow=2
        C: ancestors=2, descendants=1 → flow=2
        A: ancestors=0 → flow=0
        D: descendants=0 → flow=0
        """
        network = ReferralNetwork()
        network.add_referral("A", "B")
        network.add_referral("B", "C")
        network.add_referral("C", "D")

        result = top_k_by_flow_centrality(network, 4)
        # B and C both have flow=2, A and D have flow=0
        assert set(result[:2]) == {"B", "C"}
        assert set(result[2:]) == {"A", "D"}

    def test_middle_node_highest(self):
        """
        A → B → C → D → E
        C is in the middle: ancestors=2, descendants=2 → flow=4
        """
        network = ReferralNetwork()
        network.add_referral("A", "B")
        network.add_referral("B", "C")
        network.add_referral("C", "D")
        network.add_referral("D", "E")

        result = top_k_by_flow_centrality(network, 1)
        assert result[0] == "C"  # flow = 2 * 2 = 4

    def test_root_and_leaf_zero_flow(self):
        """Roots and leaves always have flow=0."""
        network = ReferralNetwork()
        network.add_referral("A", "B")
        network.add_referral("B", "C")

        result = top_k_by_flow_centrality(network, 3)
        # B has flow=1, A and C have flow=0
        assert result[0] == "B"

    def test_empty_network(self):
        network = ReferralNetwork()
        result = top_k_by_flow_centrality(network, 5)
        assert result == []

    def test_wide_tree_low_flow(self):
        """
        Wide tree: A → {B, C, D, E}
        All children are leaves with 0 descendants.
        A is root with 0 ancestors.
        All nodes have flow=0.
        """
        network = ReferralNetwork()
        network.add_referral("A", "B")
        network.add_referral("A", "C")
        network.add_referral("A", "D")
        network.add_referral("A", "E")

        result = top_k_by_flow_centrality(network, 5)
        # All have flow=0, order doesn't matter
        assert len(result) == 5
