SET search_path = handycapper;

-- rkm_track_offsets — per-track v0/decay offsets vs the cross-track mean,
-- inferred from the shipping-horse network in compute_adjustments.py.
-- A horse running at track X gets adj_v0 = v0 - rkm_track_offsets.v0_offset
-- to remove track-specific track-bias.
--
-- Audit RKM-T1.1 (2026-05-29) confirmed the v0_offset signs are inverted
-- relative to actual mean track speeds (BEL=0, AQU=+0.72, SAR=+0.54
-- but the actual mean ft/s is BEL > SAR > AQU). The disposition was
-- CONFIRMED + DEFERRED — within-race ranking still works because all
-- horses share the same track offset. A real fix would re-derive the
-- offsets with a canonical anchor; tracked separately.
--
-- Captured here from production state on 2026-05-29. Was V007 in the
-- original numbering plan; the slot was reserved but the migration
-- never written.

CREATE TABLE IF NOT EXISTS rkm_track_offsets (
    track        varchar(5) PRIMARY KEY,
    v0_offset    numeric(5, 2) NOT NULL,
    decay_offset numeric(6, 4),
    n_shippers   integer,
    confidence   numeric(3, 2)
);
