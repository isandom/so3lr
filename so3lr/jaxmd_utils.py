import jax
import jax.numpy as jnp

from functools import partial
from jax_md import partition
from jax_md.space import DisplacementOrMetricFn, Box, transform

from so3lr.graph import Graph


def neighbor_list_featurizer(displacement_fn, species, total_charge=0., num_unpaired_electrons=0., dtype=jnp.float32, fractional_coordinates=True):
    def featurize(R, neighbor, neighbor_lr, **kwargs):
        idx_i = neighbor[0]  # shape: P
        idx_j = neighbor[1]  # shape: P
        idx_i_lr = neighbor_lr[0]  # shape: P
        idx_j_lr = neighbor_lr[1]  # shape: P

        Ra = R[idx_i]
        Rb = R[idx_j]
        Ra_lr = R[idx_i_lr]
        Rb_lr = R[idx_j_lr]

        box = None
        positions = None
        k_smearing = None
        k_grid = None
        if 'box' in kwargs:
            box = kwargs.get('box')
            if box is not None:
                if box.shape == (3, 3):
                    box = box
                elif box.shape == (3, ):
                    box = jnp.diag(box)
                elif box.shape == (1,):
                    box = jnp.diag(jnp.repeat(box, 3))
                else:
                    raise ValueError(f"k-space electrostatics: Invalid box shape {box.shape}. Expected (3, 3), (3,), or (1,).")

        if 'k_grid' in kwargs:
            k_grid = kwargs.get('k_grid')
        if 'k_smearing' in kwargs:
            k_smearing = kwargs.get('k_smearing')

        # Resolve Cartesian positions (needed for k-space and/or perturbation)
        if k_grid is not None or 'perturbation' in kwargs:
            if fractional_coordinates:
                positions = transform(box, R)  # fractional → Cartesian
            else:
                positions = R

        if 'perturbation' in kwargs:
            pert = kwargs.get('perturbation')
            positions = transform(pert, positions)
            box = transform(pert, box)

        d = jax.vmap(partial(displacement_fn, **kwargs))
        dR = d(Ra, Rb)
        dR_lr = d(Ra_lr, Rb_lr)

        graph = Graph(
            positions=positions,
            nodes=species,
            edges=dR,
            centers=idx_i,
            others=idx_j,
            mask=None,
            total_charge=jnp.array([total_charge], dtype=dtype),
            num_unpaired_electrons=jnp.array([num_unpaired_electrons], dtype=dtype),
            edges_lr=dR_lr,
            idx_i_lr=idx_i_lr,
            idx_j_lr=idx_j_lr,
            cell=box,  # will raise an error if box not in kwargs.
            k_grid=k_grid,
            k_smearing=k_smearing,
            theory_mask=jnp.eye(16)[0].reshape(1, -1)
        )

        return graph

    return featurize


def to_jax_md(
        potential,  # the mlff potential
        displacement_or_metric: DisplacementOrMetricFn,
        box_size: Box,  # box if it exists, check md jax documentation for conventions
        species: jnp.ndarray = None,  # the atomic species, jnp.ndarray of shape n_atoms
        dr_threshold: float = 0.,  # currently dr_threshold > 0 is experimental
        capacity_multiplier: float = 1.25,
        buffer_size_multiplier_sr: float = 1.25,
        buffer_size_multiplier_lr: float = 1.25,
        minimum_cell_size_multiplier_sr: float = 1.0,
        minimum_cell_size_multiplier_lr: float = 1.0,
        disable_cell_list: bool = False,
        fractional_coordinates: bool = True,
        total_charge: float = 0.,
        num_unpaired_electrons: float = 0.,
        dtype: jnp.dtype = jnp.float32,
        **neighbor_kwargs
):
    # create the neighbor_fn
    neighbor_fn = partition.neighbor_list(
        displacement_or_metric,
        box_size,
        potential.cutoff,  # load the cutoff of the model from the MLFFPotential
        dr_threshold,
        capacity_multiplier,
        buffer_size_multiplier_sr,  # as buffer_size_multiplier
        minimum_cell_size_multiplier_sr,
        fractional_coordinates=fractional_coordinates,
        format=partition.NeighborListFormat(1),  # only sparse is supported in mlff
        disable_cell_list=disable_cell_list,
        **neighbor_kwargs)

    # create the neighbor_fn for long-range cutoff
    neighbor_fn_lr = partition.neighbor_list(
        displacement_or_metric,
        box_size,
        potential.long_range_cutoff,
        dr_threshold,
        capacity_multiplier,
        buffer_size_multiplier_lr,  # as buffer_size_multiplier
        minimum_cell_size_multiplier_lr,
        fractional_coordinates=fractional_coordinates,
        format=partition.NeighborListFormat(2),  # long-range modules can handle OrderedSparse.
        disable_cell_list=disable_cell_list,
        **neighbor_kwargs)

    featurizer = neighbor_list_featurizer(
        displacement_or_metric,
        species,
        total_charge=total_charge,
        num_unpaired_electrons=num_unpaired_electrons,
        dtype=dtype,
        fractional_coordinates=fractional_coordinates,
    )

    # create an energy_fn that is compatible with jax_md
    def energy_fn(
            R,
            neighbor,
            neighbor_lr,
            has_aux=False,
            **energy_fn_kwargs
    ):
        graph = featurizer(R, neighbor, neighbor_lr, **energy_fn_kwargs)
        if has_aux:
            return potential(graph, has_aux=True)
        else:
            return potential(graph).sum()

    return neighbor_fn, neighbor_fn_lr, energy_fn
