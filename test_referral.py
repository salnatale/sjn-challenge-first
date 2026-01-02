import pytest
from main import ReferralNetwork, ReferralError


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
