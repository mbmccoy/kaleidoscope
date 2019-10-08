import math
from itertools import chain
from numbers import Real
from typing import Tuple, Dict, Optional

import networkx as nx
import numpy as np


class CoxeterGraph:

    """
    Representation of a Coxeter graph.
    """

    def __init__(
        self,
        definition: Dict[Tuple[int, int], Real],
        dimension: Optional[int] = None,
    ):
        all_vertices = set(chain(*(k for k in definition.keys())))
        min_val = min(all_vertices)
        max_val = max(all_vertices)

        if min_val < 0:
            raise ValueError(
                "Vertices must be indexed with ints starting at zero"
            )
        if dimension is not None and max_val > dimension:
            raise ValueError(
                f"Number of vertices {max_val} greater "
                f"than dimension {dimension}"
            )

        self._dimension: int = dimension if dimension is not None else max_val

        for k, v in definition.items():
            start, stop = k
            if (stop, start) in definition and definition[(stop, start)] != v:
                raise ValueError(
                    f"Multiple values given for edge {k}: {v} and "
                    f"{definition[(stop, start)]}."
                )

        self._graph = nx.Graph()
        for edge in definition.keys():
            value = definition[edge]
            if value != 2:
                self._graph.add_edge(*edge, weight=value)

    @property
    def symmetry(self) -> Tuple[Real, ...]:
        indices = zip(*np.triu_indices(self.dimension, 1))
        return tuple(
            self._graph.edges.get(i, {"weight": 2})["weight"] for i in indices
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @classmethod
    def from_symmetry(cls, *symmetry: Real) -> "CoxeterGraph":
        discriminant = math.sqrt(1 + 8 * len(symmetry))
        dimension_test = 1 / 2 * (1 + discriminant)
        if abs(dimension_test - round(dimension_test)) > 1e-9:
            raise ValueError("Symmetry must have length d*(d-1)/2 for some d")
        dimension = round(dimension_test)
        rows, columns = np.triu_indices(dimension, 1)
        return cls(dict(zip(zip(rows, columns), symmetry)), dimension=dimension)


class FiniteReflectionGroup:

    """
    Three-dimensional finite reflection group
    """

    def __init__(self, coxeter_graph: CoxeterGraph):
        """
        """
        self._coxeter_graph = coxeter_graph
        self._dimension = coxeter_graph.dimension
        self._dihedral_angles = np.array(
            [np.cos(np.pi/order) for order in coxeter_graph.symmetry]
        )

        # Dihedral angle x implies normal angle (pi - x)
        self._dual_gramian = np.eye(coxeter_graph.dimension)
        self._dual_gramian[np.tril_indices(coxeter_graph.dimension, -1)] = \
            -self._dihedral_angles
        self._dual_gramian[np.triu_indices(coxeter_graph.dimension, 1)] = \
            -self._dihedral_angles

        try:
            self._root_system = np.linalg.cholesky(self._dual_gramian).T[:, ::-1]
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Invalid symmetry {coxeter_graph.symmetry}") from e

        u_inv = np.linalg.inv(self._root_system).T
        column_norms = np.sqrt((u_inv*u_inv).sum(0))
        self._edges = u_inv / column_norms

        # Clip to ensure roundoff error does not affect np.arccos
        self._primal_gramian = np.clip(
            self._edges.T @ self._edges, -1, 1)

    @property
    def symmetry(self) -> Tuple[Real, ...]:
        return self._coxeter_graph.symmetry

    @property
    def root_system(self) -> np.ndarray:
        """Return a root system of as columns of an array"""
        return self._root_system

    @property
    def edges(self) -> np.ndarray:
        """The 'dual' root system that defines the edges of the chamber"""
        return self._edges

    @property
    def dimension(self) -> int:
        return self._coxeter_graph.dimension

    @property
    def base_angles(self) -> np.ndarray:
        return self._primal_gramian[np.triu_indices(self.dimension, 1)].copy()

    @classmethod
    def from_symmetry(cls, *symmetry: Real) -> "FiniteReflectionGroup":
        return cls(CoxeterGraph.from_symmetry(*symmetry))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.symmetry})"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, FiniteReflectionGroup):
            return False
        return self.symmetry == other.symmetry
