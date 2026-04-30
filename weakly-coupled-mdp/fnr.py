"""
Feasible network relaxation (FNR) utilities for weakly coupled MDPs.

This module provides two main capabilities:

1. build a layered network representing feasible joint actions under the
   linking constraints;
2. solve the corresponding FNR linear program when ``gurobipy`` is available.

Typical usage:

``network = construct_fnr_network(wmdp.linking_constraints)``
``objective, x_values, policy = solve_fnr(wmdp, network)``
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from policy import ActionVector, Policy
from wmdp import LinkingConstraints, StateComponent, StateSpace, WMDP

import matplotlib.pyplot as plt
import networkx as nx
import sys

import gurobipy as gp
from gurobipy import GRB


FlowKey = Tuple[int, int, object, int]
NetworkFlowKey = Tuple[int, int, object, int]


class FNRPolicy(Policy):
    """Feasible policy extracted from an FNR solution.

    The policy follows the solved network-arc flows. Given a joint state, it
    starts at the root of the FNR network and repeatedly selects an outgoing
    arc with maximum positive flow for the current component state. If the
    path reaches a node with no positive outgoing flow, the policy falls back
    to a feasible joint action that keeps the chosen prefix and maximizes the
    number of action-1 decisions.
    """

    network_flows: Dict[NetworkFlowKey, float]
    linking_constraints: LinkingConstraints
    network: "Network"
    tolerance: float

    def __init__(
        self,
        network_flows: Dict[NetworkFlowKey, float],
        linking_constraints: LinkingConstraints,
        network: "Network",
        tolerance: float = 1e-9,
    ) -> None:
        """Initialize an FNR policy from solved network-arc flows."""
        self.network_flows = dict(network_flows)
        self.linking_constraints = linking_constraints
        self.network = network
        self.tolerance = tolerance

    def _fallback_action(self, prefix: Sequence[int], current_node: "Node") -> ActionVector:
        """Complete the current network path greedily with as many ones as possible."""
        action = list(prefix)
        node = current_node

        while len(action) < self.linking_constraints.J:
            if not node.out_arcs:
                raise RuntimeError(
                    "The FNR network path ended before all component actions were selected."
                )

            chosen_arc = max(node.out_arcs, key=lambda arc: (arc.action == 1, arc.action))
            action.append(chosen_arc.action)
            node = chosen_arc.target

        return tuple(action)

    def get_action(self, period: int, state: Sequence[StateComponent]) -> ActionVector:
        """Return a feasible joint action for the given period and joint state."""
        if len(state) != self.linking_constraints.J:
            raise ValueError("The number of component states must match the number of components.")

        if self.network.root is None:
            raise RuntimeError("The FNR network has no root node.")

        action_prefix: List[int] = []
        current_node = self.network.root
        for component, component_state in enumerate(state):
            positive_arcs = []
            for arc in current_node.out_arcs:
                flow_value = self.network_flows.get(
                    (period, component, component_state.label, arc.index),
                    0.0,
                )
                if flow_value > self.tolerance:
                    positive_arcs.append((flow_value, arc))

            if not positive_arcs:
                return self._fallback_action(action_prefix, current_node)

            _, chosen_arc = max(
                positive_arcs,
                key=lambda item: (item[0], item[1].action),
            )
            action_prefix.append(chosen_arc.action)
            current_node = chosen_arc.target

        return tuple(action_prefix)


@dataclass
class FNRResult:
    """Solution returned by the FNR optimization model."""

    objective_value: float
    marginal_flows: Dict[FlowKey, float]
    network_flows: Dict[NetworkFlowKey, float]
    policy: FNRPolicy



class Node:
    """Node in the layered FNR network.

    Parameters
    ----------
    layer
        Network layer index.
    index
        Node index within the layer.
    state
        Accumulated resource-consumption state induced by past actions.
    """

    layer: int
    index: int
    u: Tuple[float, ...]
    in_arcs: List["Arc"]
    out_arcs: List["Arc"]

    def __init__(self, layer: int, index: int, state: Sequence[float]) -> None:
        """Initialize one network node."""
        self.layer = layer
        self.index = index
        self.u = tuple(state)
        self.in_arcs = []
        self.out_arcs = []

    def __repr__(self) -> str:
        """Return a compact representation for debugging."""
        return f"Node(layer={self.layer}, index={self.index}, state={self.u})"


class Arc:
    """Arc in the layered FNR network.

    Parameters
    ----------
    layer
        Origin layer of the arc.
    index
        Global arc index in the network.
    action
        Component action associated with the arc.
    source
        Source node.
    target
        Target node.
    """

    layer: int
    index: int
    action: int
    source: Node
    target: Node
    grb_var: Optional["gp.Var"]

    def __init__(self, layer: int, index: int, action: int, source: Node, target: Node) -> None:
        """Initialize one network arc."""
        self.layer = layer
        self.index = index
        self.action = action
        self.source = source
        self.target = target
        self.grb_var = None

    def __repr__(self) -> str:
        """Return a compact representation for debugging."""
        return (
            f"Arc(layer={self.layer}, index={self.index}, action={self.action}, "
            f"source={self.source.index}, target={self.target.index})"
        )


class Network:
    """Layered network used by the feasible network relaxation.

    Parameters
    ----------
    num_layers
        Number of network layers. This should typically be ``J + 1`` where
        ``J`` is the number of components.
    """

    L: int
    layers: List[Dict[Tuple[float, ...], Node]]
    root: Optional[Node]

    def __init__(self, num_layers: int) -> None:
        """Initialize an empty layered network."""
        self.L = num_layers
        self.layers = [{} for _ in range(num_layers)]
        self.root = None

    def print(self) -> None:
        """Print a readable description of the network."""
        for layer in range(self.L):
            print("Layer:", layer)
            for key, node in self.layers[layer].items():
                print("\tNode:", key, "u:", node.u, "index:", node.index)
                for arc in node.out_arcs:
                    print(
                        "\t\tArc:",
                        arc.index,
                        "action:",
                        arc.action,
                        "target:",
                        arc.target.index,
                        "target u:",
                        arc.target.u,
                    )


    def reduce(self) -> None:
        """Merge nodes with identical continuation structure.

        This is the standard backward reduction step used to shrink the network
        after it has been built from the linking constraints.
        """
        for layer in range(self.L - 1, 0, -1):
            new_layer: Dict[Tuple[Tuple[int, int], ...], Node] = {}
            for node_candidate in self.layers[layer].values():
                reduced_state = tuple(
                    sorted((arc.action, arc.target.index) for arc in node_candidate.out_arcs)
                )

                existing_node = new_layer.get(reduced_state)
                if existing_node is None:
                    new_layer[reduced_state] = node_candidate
                else:
                    for arc in node_candidate.in_arcs:
                        arc.target = existing_node
                        existing_node.in_arcs.append(arc)

                    for arc in node_candidate.out_arcs:
                        if arc in arc.target.in_arcs:
                            arc.target.in_arcs.remove(arc)

            self.layers[layer] = {node.u: node for node in new_layer.values()}


    def construct_network(self, linkct: LinkingConstraints) -> None:
        """Construct the feasibility network from linking constraints.

        The resulting layered network preserves the component-by-component flow
        interpretation while restricting flow to paths that correspond to
        feasible joint actions.
        """
        for k in range(linkct.K):
            for j in range(linkct.J):
                for action in linkct.A[j]:
                    if linkct.C[(k, j, action)] < 0:
                        raise ValueError(
                            "FNR network construction requires nonnegative linking coefficients."
                        )

        num_arcs = 0
        root_state = tuple(0.0 for _ in range(linkct.K))
        root = Node(layer=0, index=0, state=root_state)
        self.layers[0][root_state] = root
        self.root = root

        for component in range(self.L - 1):
            for node in list(self.layers[component].values()):
                for action in linkct.A[component]:
                    next_state = list(node.u)
                    feasible = True

                    for k in range(linkct.K):
                        next_state[k] += linkct.C[(k, component, action)]
                        if next_state[k] > linkct.b[k]:
                            feasible = False
                            break

                    if not feasible:
                        continue

                    next_state_tuple = tuple(next_state)
                    target_node = self.layers[component + 1].get(next_state_tuple)
                    if target_node is None:
                        target_node = Node(
                            layer=component + 1,
                            index=len(self.layers[component + 1]),
                            state=next_state_tuple,
                        )
                        self.layers[component + 1][next_state_tuple] = target_node

                    arc = Arc(
                        layer=component,
                        index=num_arcs,
                        action=action,
                        source=node,
                        target=target_node,
                    )
                    node.out_arcs.append(arc)
                    target_node.in_arcs.append(arc)
                    num_arcs += 1

        self.reduce()

    def get_size(self) -> Dict[str, int]:
        """Return basic size statistics for the network."""
        num_nodes = 0
        num_arcs = 0
        max_width = 0
        for layer in range(self.L):
            num_nodes += len(self.layers[layer])
            max_width = max(max_width, len(self.layers[layer]))
            for node in self.layers[layer].values():
                num_arcs += len(node.out_arcs)
        return {"nodes": num_nodes, "arcs": num_arcs, "max_width": max_width}


def construct_fnr_network(linking_constraints: LinkingConstraints) -> Network:
    """Build and return the FNR network for a given linking-constraint system.

    Parameters
    ----------
    linking_constraints
        Linking constraints defining the feasible joint-action set.

    Returns
    -------
    Network
        A reduced layered network with ``J + 1`` layers.

    Example
    -------
    ``network = construct_fnr_network(wmdp.linking_constraints)``
    """
    network = Network(num_layers=linking_constraints.J + 1)
    network.construct_network(linking_constraints)
    return network


def draw_fnr_network(network: Network, figsize: Tuple[int, int] = (10, 5)) -> None:
    """Draw the FNR network with nodes arranged by layer.

    Parameters
    ----------
    network
        FNR feasibility network.
    figsize
        Matplotlib figure size.

    Notes
    -----
    This function requires ``matplotlib`` and ``networkx``.
    """
    if plt is None or nx is None:
        raise ImportError(
            "draw_fnr_network requires matplotlib and networkx."
        )

    graph = nx.DiGraph()
    positions = {}
    node_labels = {}
    edge_actions = {}

    for layer in range(network.L):
        nodes_in_layer = sorted(network.layers[layer].values(), key=lambda node: node.index)
        layer_size = len(nodes_in_layer)
        vertical_offset = (layer_size - 1) / 2.0

        for row, node in enumerate(nodes_in_layer):
            node_id = (node.layer, node.index)
            graph.add_node(node_id)
            positions[node_id] = (node.layer, vertical_offset - row)
            node_labels[node_id] = f"v{node.layer + 1},{node.index + 1}\n{node.u}"

            for arc in node.out_arcs:
                source_id = (arc.source.layer, arc.source.index)
                target_id = (arc.target.layer, arc.target.index)
                graph.add_edge(source_id, target_id)
                edge_actions.setdefault((source_id, target_id), []).append(arc.action)

    figure, axis = plt.subplots(figsize=figsize)
    nx.draw(
        graph,
        pos=positions,
        with_labels=False,
        node_color="#f4f1de",
        node_size=2200,
        edge_color="#3d405b",
        arrows=True,
    )
    nx.draw_networkx_labels(graph, pos=positions, labels=node_labels, font_size=9)
    nx.draw_networkx_edge_labels(
        graph,
        pos=positions,
        edge_labels={
            edge: ",".join(str(action) for action in sorted(actions))
            for edge, actions in edge_actions.items()
        },
        font_color="#e07a5f",
    )

    if positions:
        y_values = [position[1] for position in positions.values()]
        min_y = min(y_values)
        max_y = max(y_values)
    else:
        min_y = 0.0
        max_y = 0.0

    label_height = max_y + 0.55
    for layer in range(network.L):
        if layer == 0:
            layer_name = "Source"
        elif layer == network.L - 1:
            layer_name = "Terminal"
        else:
            layer_name = f"Component {layer}"

        plt.text(
            layer,
            label_height,
            layer_name,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    axis.set_title("FNR feasibility network", pad=8)
    axis.set_xlim(-0.45, network.L - 0.55)
    axis.set_ylim(min_y - 0.55, label_height + 0.35)
    axis.axis("off")
    figure.tight_layout()
    plt.show()


class FNR:
    """
    Otimization model for the feasible network relaxation.
    """

    def __init__(self, wmdp: WMDP, network: Network) -> None:
        """Build the FNR linear program for the provided WMDP and network."""
        self.wmdp = wmdp
        self.network = network
        self.state_space: StateSpace = wmdp.state_space
        self.linkct = wmdp.linking_constraints

        self.J = wmdp.J
        self.T = wmdp.T
        self.Jset = range(self.J)
        self.Tset = range(self.T)

        self.model = gp.Model("fnr")
        self.x = None
        self.y = None

        self._build_model()
        self.model.setParam("OutputFlag", 0)
        self.model.setParam("FeasibilityTol", 1e-9)


    def _build_model(self) -> None:
        """
        Create the full FNR linear program.
        """
        
        # Variables (x)
        flow_index = []
        for t in self.Tset:
            for j in self.Jset:
                for state in self.state_space.S[j].states[t]:
                    for action in self.state_space.A[j]:
                        flow_index.append((t, j, state.label, action))
        self.x = self.model.addVars(flow_index, name="x", lb=0.0)

        # Initial states
        for j in self.Jset:
            self.model.addConstr(
                gp.quicksum(
                    self.x[0, j, state.label, action]
                    for state in self.state_space.S[j].states[0]
                    for action in self.state_space.A[j]
                )
                == 1,
                name=f"initial_flow_{j}",
            )

        # Conservation of flow across periods
        for t in range(1, self.T):
            for j in self.Jset:
                for state in self.state_space.S[j].states[t]:
                    self.model.addConstr(
                        gp.quicksum(
                            self.x[t, j, state.label, action]
                            for action in self.state_space.A[j]
                        )
                        == gp.quicksum(
                            self.state_space.S[j].P[t - 1].get(
                                (prev_state.label, action, state.label),
                                0.0,
                            )
                            * self.x[t - 1, j, prev_state.label, action]
                            for prev_state in self.state_space.S[j].states[t - 1]
                            for action in self.state_space.A[j]
                        ),
                        name=f"state_flow_{t}_{j}_{state.label}",
                    )

        # Network variables
        network_flow_index = []
        for t in self.Tset:
            for layer in range(self.network.L - 1):
                for node in self.network.layers[layer].values():
                    for state in self.state_space.S[layer].states[t]:
                        for arc in node.out_arcs:
                            network_flow_index.append((t, arc.layer, state.label, arc.index))
        self.y = self.model.addVars(network_flow_index, name="y", lb=0.0)

        # Network flows
        for t in self.Tset:
            for j in self.Jset:
                for state in self.state_space.S[j].states[t]:
                    for action in self.state_space.A[j]:
                        self.model.addConstr(
                            gp.quicksum(
                                self.y[t, j, state.label, arc.index]
                                for node in self.network.layers[j].values()
                                for arc in node.out_arcs
                                if arc.action == action
                            )
                            == self.x[t, j, state.label, action],
                            name=f"link_{t}_{j}_{state.label}_{action}",
                        )

            for layer in range(1, self.network.L - 1):
                for node in self.network.layers[layer].values():
                    self.model.addConstr(
                        gp.quicksum(
                            self.y[t, arc.layer, state.label, arc.index]
                            for arc in node.out_arcs
                            for state in self.state_space.S[arc.layer].states[t]
                        )
                        == gp.quicksum(
                            self.y[t, arc.layer, state.label, arc.index]
                            for arc in node.in_arcs
                            for state in self.state_space.S[arc.layer].states[t]
                        ),
                        name=f"network_balance_{t}_{layer}_{node.index}",
                    )

        # Objective
        self.model.setObjective(
            gp.quicksum(
                state.reward[action] * self.x[t, j, state.label, action]
                for t in self.Tset
                for j in self.Jset
                for state in self.state_space.S[j].states[t]
                for action in self.state_space.A[j]
            ),
            GRB.MAXIMIZE,
        )


    def _extract_marginal_flows(self) -> Dict[FlowKey, float]:
        """Return the current optimal marginal flow values."""
        marginal_flows: Dict[FlowKey, float] = {}
        for t in self.Tset:
            for j in self.Jset:
                for state in self.state_space.S[j].states[t]:
                    for action in self.state_space.A[j]:
                        marginal_flows[(t, j, state.label, action)] = self.x[
                            t, j, state.label, action
                        ].X
        return marginal_flows


    def _extract_network_flows(self) -> Dict[NetworkFlowKey, float]:
        """Return the current optimal flow values on FNR network arcs."""
        network_flows: Dict[NetworkFlowKey, float] = {}
        for t in self.Tset:
            for layer in range(self.network.L - 1):
                for node in self.network.layers[layer].values():
                    for state in self.state_space.S[layer].states[t]:
                        for arc in node.out_arcs:
                            network_flows[(t, layer, state.label, arc.index)] = self.y[
                                t, layer, state.label, arc.index
                            ].X
        return network_flows


    def optimize(self) -> FNRResult:
        """Solve the current FNR model and return the resulting policy data."""
        print("Optimizing...")
        self.model.optimize()

        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"FNR solve did not terminate optimally. Gurobi status: {self.model.Status}"
            )

        marginal_flows = self._extract_marginal_flows()
        network_flows = self._extract_network_flows()

        print("\tdone.")
        print(
            "FNR model size:",
            f"vars={self.model.NumVars}",
            f"constrs={self.model.NumConstrs}",
            f"nonzeros={self.model.NumNZs}",
            f"time={self.model.Runtime}s",
        )

        policy = FNRPolicy(
            network_flows=network_flows,
            linking_constraints=self.linkct,
            network=self.network,
        )
        return FNRResult(
            objective_value=self.model.ObjVal,
            marginal_flows=marginal_flows,
            network_flows=network_flows,
            policy=policy,
        )

def solve_fnr(
    wmdp: WMDP,
    network: Network,
) -> Tuple[float, Dict[Tuple[int, int, object, int], float], FNRPolicy]:
    """Build and solve the feasible network relaxation LP.

    Parameters
    ----------
    wmdp
        Weakly coupled MDP instance to be relaxed.
    network
        Feasibility network built from the linking constraints.

    Returns
    -------
    tuple
        A triple ``(objective_value, marginal_flows, policy)`` where
        ``marginal_flows[(t, j, s, a)]`` stores the optimal marginal flow and
        ``policy`` is a feasible policy extracted from those flows.

    Example
    -------
    ``objective, marginal_flows, policy = solve_fnr(wmdp, network)``

    Notes
    -----
    This routine requires ``gurobipy`` to be installed and licensed.
    """

    fnr_model = FNR(wmdp, network)
    result = fnr_model.optimize()
    return result.objective_value, result.marginal_flows, result.policy
