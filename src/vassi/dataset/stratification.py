import numpy as np


def stratified_subsample(
    strata_labels: np.ndarray,
    *,
    size: int,
    random_state: int | np.random.Generator | None,
) -> np.ndarray:
    """
    Performs stratified subsampling using the Largest Remainder Method.
    """
    population_size = len(strata_labels)
    if size < 0:
        raise ValueError("size cannot be negative.")

    if population_size == 0:
        raise ValueError("Cannot sample from an empty population (N=0).")

    if size == 0:
        return np.array([], dtype=int)

    if size > population_size:
        raise ValueError(
            f"Cannot request size={size} which is larger than the total population size N={population_size}."
        )
    if size == population_size:
        return np.arange(population_size)

    random_state = np.random.default_rng(random_state)

    # 1. Get unique strata and their counts
    _, strata_labels_inverse, original_strata_counts = np.unique(
        strata_labels, return_inverse=True, return_counts=True
    )
    n_strata = len(original_strata_counts)

    # 2. Calculate ideal counts
    ideal_sample_counts_float = (original_strata_counts / population_size) * size

    # 3. Use LRM
    final_sample_counts_int = np.floor(ideal_sample_counts_float).astype(int)
    remainders = ideal_sample_counts_float - final_sample_counts_int
    n_allocated = final_sample_counts_int.sum()
    n_remaining_to_allocate = size - n_allocated

    if n_remaining_to_allocate > 0:
        random_tiebreakers = random_state.random(n_strata)
        indices_sorted = np.lexsort((random_tiebreakers, remainders))
        indices_to_increment = indices_sorted[-n_remaining_to_allocate:]
        final_sample_counts_int[indices_to_increment] += 1

    # 4. Safety checks
    if final_sample_counts_int.sum() != size:
        raise RuntimeError("Largest Remainder Method failed to sum correctly.")
    if np.any(final_sample_counts_int > original_strata_counts):
        raise RuntimeError("Sampling algorithm trying to sample more than available.")

    # 5. Perform vectorized sampling
    original_indices = np.arange(population_size)
    random_values_for_sorting = random_state.random(population_size)

    sort_keys = np.lexsort((random_values_for_sorting, strata_labels_inverse))
    shuffled_indices_grouped_by_stratum = original_indices[sort_keys]

    stratum_start_indices = np.cumsum(np.r_[0, original_strata_counts[:-1]])

    sorted_strata_ids = strata_labels_inverse[sort_keys]
    mapped_start_indices = stratum_start_indices[sorted_strata_ids]
    group_relative_counter = np.arange(population_size) - mapped_start_indices

    sample_counts_per_item = final_sample_counts_int[sorted_strata_ids]

    mask = group_relative_counter < sample_counts_per_item
    final_sample = shuffled_indices_grouped_by_stratum[mask]
    final_sample.sort()

    return final_sample


def biased_stratified_subsample(
    strata_labels: np.ndarray,
    *,
    size: int,
    min_samples_per_stratum: int,
    random_state: int | np.random.Generator | None,
) -> np.ndarray:
    """
    Performs biased stratified subsampling.

    This function guarantees a `min_samples_per_stratum` "floor" for each
    stratum (respecting original counts) and then fills the remaining
    `size` proportionally using the Largest Remainder Method.

    If the total "floor" exceeds `size` (Case A), it proportionally
    down-samples the floor allocation to fit the `size` budget.
    """
    population_size = len(strata_labels)
    if size < 0:
        raise ValueError("size cannot be negative.")
    if min_samples_per_stratum < 0:
        raise ValueError("min_samples_per_stratum cannot be negative.")

    if population_size == 0:
        raise ValueError("Cannot sample from an empty population (N=0).")

    if size == 0:
        return np.array([], dtype=int)

    if size > population_size:
        raise ValueError(
            f"Cannot request size={size} which is larger than the total population size N={population_size}."
        )
    if size == population_size:
        return np.arange(population_size)

    random_state = np.random.default_rng(random_state)

    # 1. Get original strata info
    _, strata_labels_inverse, original_strata_counts = np.unique(
        strata_labels, return_inverse=True, return_counts=True
    )
    n_strata = len(original_strata_counts)

    # 2. STAGE 1: Calculate Guaranteed Allocation (The "Floor")
    guaranteed_counts = np.minimum(min_samples_per_stratum, original_strata_counts)
    total_guaranteed_size = guaranteed_counts.sum()

    # 3. STAGE 2: Allocate Remainder
    if total_guaranteed_size > size:
        # --- CASE A: Over-Subscribed ---
        # Proportionally allocate `size` using `guaranteed_counts` as weights

        proportions = guaranteed_counts / total_guaranteed_size
        ideal_sample_counts_float = proportions * size

        final_sample_counts_int = np.floor(ideal_sample_counts_float).astype(int)
        remainders = ideal_sample_counts_float - final_sample_counts_int
        n_allocated = final_sample_counts_int.sum()
        n_remaining = size - n_allocated

        if n_remaining > 0:
            random_tiebreakers = random_state.random(n_strata)
            indices_sorted = np.lexsort((random_tiebreakers, remainders))
            indices_to_increment = indices_sorted[-n_remaining:]
            final_sample_counts_int[indices_to_increment] += 1

    else:
        # --- CASE B: Happy Path (total_guaranteed_size <= size) ---
        remaining_size_to_fill = size - total_guaranteed_size

        if remaining_size_to_fill == 0:
            # We are done, the floor *is* the final allocation
            final_sample_counts_int = guaranteed_counts

        else:
            # Define the "remaining" population to sample from
            remaining_strata_counts = original_strata_counts - guaranteed_counts
            remaining_population_size = remaining_strata_counts.sum()

            if remaining_population_size == 0:
                # This happens if total_guaranteed_size == population_size
                # (and size > population_size, which is impossible due to guards)
                # Or if min_samples_per_stratum >= max(counts)
                # We are done, the floor is all we can take.
                final_sample_counts_int = guaranteed_counts

            else:
                # Run LRM on the remaining population
                ideal_fill_counts_float = (
                    remaining_strata_counts / remaining_population_size
                ) * remaining_size_to_fill

                proportional_fill_counts = np.floor(ideal_fill_counts_float).astype(int)
                remainders = ideal_fill_counts_float - proportional_fill_counts
                n_allocated = proportional_fill_counts.sum()
                n_remaining = remaining_size_to_fill - n_allocated

                if n_remaining > 0:
                    random_tiebreakers = random_state.random(n_strata)
                    # We must mask out strata with 0 remaining counts
                    remainders[remaining_strata_counts == 0] = -1
                    indices_sorted = np.lexsort((random_tiebreakers, remainders))
                    indices_to_increment = indices_sorted[-n_remaining:]
                    proportional_fill_counts[indices_to_increment] += 1

                # Combine the floor + proportional fill
                final_sample_counts_int = guaranteed_counts + proportional_fill_counts

    # 4. Perform safety checks
    if final_sample_counts_int.sum() != size:
        raise RuntimeError("Biased LRM failed to sum to `size` correctly.")
    if np.any(final_sample_counts_int > original_strata_counts):
        raise RuntimeError(
            "Biased sampling algorithm trying to sample more than available."
        )

    # 5. Perform vectorized sampling (this logic is identical to the original)
    original_indices = np.arange(population_size)
    random_values_for_sorting = random_state.random(population_size)

    sort_keys = np.lexsort((random_values_for_sorting, strata_labels_inverse))
    shuffled_indices_grouped_by_stratum = original_indices[sort_keys]

    stratum_start_indices = np.cumsum(np.r_[0, original_strata_counts[:-1]])

    sorted_strata_ids = strata_labels_inverse[sort_keys]
    mapped_start_indices = stratum_start_indices[sorted_strata_ids]
    group_relative_counter = np.arange(population_size) - mapped_start_indices

    sample_counts_per_item = final_sample_counts_int[sorted_strata_ids]

    mask = group_relative_counter < sample_counts_per_item
    final_sample = shuffled_indices_grouped_by_stratum[mask]
    final_sample.sort()

    return final_sample
