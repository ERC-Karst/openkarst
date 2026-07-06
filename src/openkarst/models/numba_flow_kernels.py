"""Numba kernels for conduit flow updates."""

import math

from openkarst.models.numba_support import (
    NUMBA_AVAILABLE,
    ensure_numba_available,
    njit,
    prange,
)


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _churchill_friction_factor_scalar(reynolds, roughness, diameter):
        c = (7.0 / reynolds) ** 0.9 + 0.27 * roughness / diameter
        a = (-2.457 * math.log(c)) ** 16.0
        b = (37530.0 / reynolds) ** 16.0
        return 8.0 * ((8.0 / reynolds) ** 12.0 + 1.0 / ((a + b) ** 1.5)) ** (
            1.0 / 12.0
        )


    @njit(cache=True)
    def _divide_like_numpy(numerator, denominator):
        if denominator != 0.0:
            return numerator / denominator
        if numerator > 0.0:
            return math.inf
        if numerator < 0.0:
            return -math.inf
        return math.nan


    @njit(parallel=True, cache=True)
    def compute_flow_update_numba(
        Q_new,
        Q_old_t,
        Q_prev_i,
        a1,
        a2,
        a_mid_new,
        a_mid_old_t,
        r1,
        r2,
        r_mid,
        h1,
        h2,
        v_mid,
        froude,
        alpha,
        w_mid,
        is_full_y_mid,
        full_conduit_areas,
        full_hydraulic_diameters,
        conduit_epsilon,
        conduit_manning,
        bc_flux_node,
        n_indices1,
        n_indices2,
        f,
        dQ_friction,
        q_correction,
        D_eff,
        Re_conduit,
        conduit_lengths,
        gravity,
        water_density,
        dynamic_viscosity,
        dt,
        relaxation_factor,
        geometry_channel,
        friction_model_churchill,
    ):
        for k in prange(Q_new.size):
            # Midpoint velocity
            if geometry_channel:
                velocity = _divide_like_numpy(Q_prev_i[k], a_mid_new[k])
            elif is_full_y_mid[k]:
                velocity = _divide_like_numpy(Q_prev_i[k], full_conduit_areas[k])
            else:
                velocity = _divide_like_numpy(Q_prev_i[k], a_mid_new[k])
            v_mid[k] = velocity

            # Froude number and upstream-weighting factor
            abs_velocity = abs(velocity)
            froude_denominator = math.sqrt(
                gravity * _divide_like_numpy(a_mid_new[k], w_mid[k])
            )
            froude_value = _divide_like_numpy(abs_velocity, froude_denominator)
            froude[k] = froude_value

            if is_full_y_mid[k]:
                alpha_value = 0.0
            elif froude_value <= 0.5:
                alpha_value = 1.0
            elif froude_value < 1.0:
                alpha_value = 2.0 * (1.0 - froude_value)
            else:
                alpha_value = 0.0
            alpha[k] = alpha_value

            # Upstream-weighted hydraulic radius and area
            if h1[k] > h2[k]:
                r_mid_upwtd = r1[k] + alpha_value * (r_mid[k] - r1[k])
                a_mid_upwtd = a1[k] + alpha_value * (a_mid_new[k] - a1[k])
            else:
                r_mid_upwtd = r2[k] + alpha_value * (r_mid[k] - r2[k])
                a_mid_upwtd = a2[k] + alpha_value * (a_mid_new[k] - a2[k])

            if geometry_channel:
                a_pressure = a_mid_upwtd
            elif is_full_y_mid[k]:
                a_pressure = full_conduit_areas[k]
            else:
                a_pressure = a_mid_upwtd

            # Pressure and inertia terms
            length = conduit_lengths[k]
            dQ_pressure = (
                _divide_like_numpy(
                    -gravity * a_pressure * (h2[k] - h1[k]),
                    length,
                )
                * dt
            )

            flux_n1 = bc_flux_node[n_indices1[k]]
            flux_n2 = bc_flux_node[n_indices2[k]]
            q_corr = 0.5 * w_mid[k] * (flux_n1 + flux_n2)
            q_correction[k] = q_corr

            dQ_inertia1 = alpha_value * 2.0 * velocity * (
                a_mid_new[k] - a_mid_old_t[k] - q_corr * dt
            )
            dQ_inertia2 = (
                _divide_like_numpy(
                    alpha_value * velocity * velocity * (a2[k] - a1[k]),
                    length,
                )
                * dt
            )

            f_local = 0.0

            # Friction term
            if geometry_channel:
                d_eff = 4.0 * r_mid[k]
                D_eff[k] = d_eff
                Re_conduit[k] = water_density * abs_velocity * d_eff / dynamic_viscosity
                dQ_friction[k] = (
                    _divide_like_numpy(
                        gravity * conduit_manning[k] ** 2.0 * abs_velocity,
                        r_mid_upwtd ** (4.0 / 3.0),
                    )
                    * dt
                )

            else:
                if is_full_y_mid[k]:
                    d_eff = full_hydraulic_diameters[k]
                else:
                    d_eff = 4.0 * r_mid[k]
                D_eff[k] = d_eff
                reynolds = water_density * abs_velocity * d_eff / dynamic_viscosity
                Re_conduit[k] = reynolds

                if friction_model_churchill:
                    if reynolds <= 2300.0:
                        f_local = 64.0 / max(reynolds, 1e-12)
                    else:
                        f_local = _churchill_friction_factor_scalar(
                            reynolds,
                            conduit_epsilon[k],
                            d_eff,
                        )
                    dQ_friction[k] = (
                        _divide_like_numpy(f_local * abs_velocity, 8.0 * r_mid[k])
                        * dt
                    )

                else:
                    if is_full_y_mid[k]:
                        if reynolds <= 2300.0:
                            f_local = 64.0 / max(reynolds, 1e-12)
                        else:
                            f_local = _churchill_friction_factor_scalar(
                                reynolds,
                                conduit_epsilon[k],
                                d_eff,
                            )
                        dQ_friction[k] = (
                            _divide_like_numpy(
                                f_local * abs_velocity,
                                8.0 * r_mid[k],
                            )
                            * dt
                        )
                    else:
                        dQ_friction[k] = (
                            _divide_like_numpy(
                                gravity * conduit_manning[k] ** 2.0 * abs_velocity,
                                r_mid_upwtd ** (4.0 / 3.0),
                            )
                            * dt
                        )

            # Relaxed discharge update
            f[k] = f_local
            q_candidate = _divide_like_numpy(
                Q_old_t[k] + dQ_pressure + dQ_inertia1 + dQ_inertia2,
                1.0 + dQ_friction[k],
            )
            Q_new[k] = (
                (1.0 - relaxation_factor) * Q_prev_i[k]
                + relaxation_factor * q_candidate
            )


else:

    def compute_flow_update_numba(*args):
        """Raise a clear error when the optional Numba backend is unavailable."""
        ensure_numba_available()
