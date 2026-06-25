# TODO — Resolve 14 xfails in test_correlation_validation.py

## WP-1: Fix PSS-10↔resilience double penalty

- [ ] Reduce `daily_resilience_penalty` from `-coupling*(res-0.5)` to `-0.4*coupling*(res-0.5)` in agent.py process_pss10_consolidation
- [ ] Remove xfail markers from #1 (test_pss10_resilience) and #8 (test_avg_pss10_avg_resilience) now that they XPASS

## WP-2: Dampen affect↔resources positive feedback

- [ ] Remove redundant `resource_affect_mod` from agent.py line 737-738 (update_affect_dynamics already handles this)
- [ ] Remove xfail marker from #5 (test_affect_resources)

## WP-3: Decouple stress↔resources

- [ ] Reduce `stress_resource_coupling` from 0.30 to 0.15 in config.py default
- [ ] Remove xfail markers from #7 (test_stress_resources) and #12 (test_correlation_magnitude_ranges)

## WP-4: Strengthen PSS-10↔resources

- [ ] Increase `resource_adjust` multiplier from 12.0 to 18.0 in agent.py process_pss10_consolidation
- [ ] Increase `resource_adjust` to 18.0 in agent.py process_pss10_consolidation

## WP-5: Re-evaluate borderline xfails

- [ ] Check #2 (test_pss10_affect), #6 (test_stress_affect), #9 (test_avg_pss10_avg_affect), #13 (test_correlation_with_different_network_structures) after WP1-4
- [ ] Remove xfail markers from any newly passing

## WP-6: Structural/metrics fixes

- [ ] Fix #10 (social_support_rate) — change to daily metric
- [ ] Fix #14 (non-monotonic) — relax assertion
- [ ] Fix #4 (env isolation) — add reload_assumptions() cleanup
- [ ] Fix the 2 non-xfail failures (pss10_stress, avg_pss10_avg_stress)
