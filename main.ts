// Deterministic week-one GTA VI PS5 trophy earn-rate forecaster (Node 22, standard library only, zero network).
//
// Reads every *.in file sorted from MAKEFUN_INPUT_DIR (default /input) and writes <id>.out to
// MAKEFUN_OUTPUT_DIR (default /output) with {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95] as const;
const Z_NORM = [
  -1.6448536269514722,
  -0.6744897501960817,
  0,
  0.6744897501960817,
  1.6448536269514722,
] as const;

const LAUNCH_EPOCH_MS = Date.UTC(2026, 10, 19, 0, 0, 0); // 2026-11-19T00:00:00Z
const MS_PER_HOUR = 3_600_000;
const PINNED_MAX_QUANTILES = [99.98, 99.99, 100, 100, 100];

const SCENARIO_WEIGHTS = [0.14, 0.22, 0.26, 0.20, 0.18] as const;
const NUM_SCENARIOS = SCENARIO_WEIGHTS.length;
const TAU_EB = 0.85;
const BISECTION_STEPS = 24;

type Category = 'platinum' | 'grind_hard' | 'story_secret' | 'action_misc';

interface TrophyMeta {
  id: number;
  name?: string;
  description?: string;
  tier?: string;
  points?: number;
  secret?: boolean;
}

interface HistoryRead {
  slot?: string;
  at?: string;
  players_tracked: number;
  trophies_earned?: number;
  rates: Array<number | null>;
}

interface CaseFile {
  schema?: string;
  id?: string;
  at?: string;
  horizon_minutes: number;
  target?: string;
  levels: number[];
  trophies: TrophyMeta[];
  history: HistoryRead[];
}

interface UniqueState {
  firstMs: number;
  lastMs: number;
  effMs: number;
  u: number;
  players: number;
  rates: Array<number | null>;
}

interface WeightedOlsResult {
  valid: boolean;
  slope: number;
  intercept: number;
  resid: number;
  sumW: number;
  count: number;
}

interface ForecastRow {
  id: number;
  q: number[];
}

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const clampPct = (pct: number): number => clamp(pct, 0.01, 99.99);
const logitPct = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogitPct = (z: number): number => 100 / (1 + Math.exp(-z));
const round4 = (x: number): number => {
  const r = Math.round(x * 10_000) / 10_000;
  return r === 0 ? 0 : r;
};

const Z_FLOOR = logitPct(0.01);
const Z_CEIL = logitPct(99.99);

function safePlayers(n: unknown): number {
  return typeof n === 'number' && Number.isFinite(n) && n > 0 ? n : 1;
}

function safeRate(r: unknown): number | null {
  if (typeof r !== 'number' || !Number.isFinite(r)) return null;
  return clamp(r, 0, 100);
}

function toLogTime(ms: number): number {
  const safeMs = Number.isFinite(ms) ? ms : LAUNCH_EPOCH_MS;
  const hoursSinceLaunch = (safeMs - LAUNCH_EPOCH_MS) / MS_PER_HOUR;
  return Math.log(1 + Math.max(0.05, hoursSinceLaunch));
}

function parseReadMs(read: HistoryRead): number {
  if (read.at !== undefined) {
    const t = Date.parse(read.at);
    if (Number.isFinite(t)) return t;
  }
  if (read.slot !== undefined) {
    const t = Date.parse(read.slot);
    if (Number.isFinite(t)) return t;
  }
  return LAUNCH_EPOCH_MS;
}

function parseTargetMs(c: CaseFile, fallbackLastMs: number, horizonMinutes: number): number {
  if (c.target !== undefined) {
    const t = Date.parse(c.target);
    if (Number.isFinite(t)) return t;
  }
  if (c.at !== undefined) {
    const t = Date.parse(c.at);
    if (Number.isFinite(t)) return t + horizonMinutes * 60_000;
  }
  return fallbackLastMs + horizonMinutes * 60_000;
}

function sameState(a: HistoryRead, b: HistoryRead): boolean {
  if (safePlayers(a.players_tracked) !== safePlayers(b.players_tracked)) return false;
  const aRates = Array.isArray(a.rates) ? a.rates : [];
  const bRates = Array.isArray(b.rates) ? b.rates : [];
  if (aRates.length !== bRates.length) return false;
  for (let i = 0; i < aRates.length; i++) {
    if (safeRate(aRates[i]) !== safeRate(bRates[i])) return false;
  }
  return true;
}

/** Group consecutive identical (players_tracked, rates) reads into unique Exophase refresh states. */
function buildUniqueStates(history: HistoryRead[]): UniqueState[] {
  const states: UniqueState[] = [];
  let prevRead: HistoryRead | null = null;

  for (const read of history) {
    const ms = parseReadMs(read);
    if (prevRead !== null && sameState(prevRead, read)) {
      states[states.length - 1].lastMs = ms;
    } else {
      states.push({
        firstMs: ms,
        lastMs: ms,
        effMs: ms,
        u: 0,
        players: safePlayers(read.players_tracked),
        rates: Array.isArray(read.rates) ? read.rates : [],
      });
    }
    prevRead = read;
  }

  for (let i = 0; i < states.length; i++) {
    const effMs = i === 0 ? states[0].firstMs : 0.5 * (states[i - 1].lastMs + states[i].firstMs);
    states[i].effMs = effMs;
    states[i].u = toLogTime(effMs);
  }

  return states;
}

function weightedOLS(
  xs: Float64Array,
  ys: Float64Array,
  ws: Float64Array,
  count: number,
  defaultResid: number,
): WeightedOlsResult {
  let sumW = 0;
  let sumWX = 0;
  let sumWY = 0;
  for (let i = 0; i < count; i++) {
    const w = ws[i];
    sumW += w;
    sumWX += w * xs[i];
    sumWY += w * ys[i];
  }
  if (count < 2 || sumW <= 0 || !Number.isFinite(sumW)) {
    return { valid: false, slope: 0, intercept: 0, resid: defaultResid, sumW: 0, count };
  }
  const meanX = sumWX / sumW;
  const meanY = sumWY / sumW;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < count; i++) {
    const dx = xs[i] - meanX;
    sxx += ws[i] * dx * dx;
    sxy += ws[i] * dx * (ys[i] - meanY);
  }
  if (sxx <= 1e-12 || !Number.isFinite(sxx) || !Number.isFinite(sxy)) {
    return { valid: false, slope: 0, intercept: meanY, resid: defaultResid, sumW, count };
  }
  const slope = sxy / sxx;
  const intercept = meanY - slope * meanX;
  let resid = defaultResid;
  if (count >= 3) {
    let rss = 0;
    for (let i = 0; i < count; i++) {
      const err = ys[i] - (intercept + slope * xs[i]);
      rss += ws[i] * err * err;
    }
    const rms = Math.sqrt(Math.max(0, rss / sumW));
    resid = Number.isFinite(rms) ? Math.max(0.01, rms) : defaultResid;
  }
  return { valid: true, slope, intercept, resid, sumW, count };
}

function classifyTrophy(t: TrophyMeta): Category {
  const tier = (t.tier ?? '').toLowerCase();
  if (tier === 'platinum') return 'platinum';

  const desc = (t.description ?? '').toLowerCase();
  if (
    desc.includes('100%') ||
    desc.includes('250 ') ||
    desc.includes('100 locations') ||
    desc.includes('50 challenges') ||
    desc.includes('all ') ||
    desc.includes('every ') ||
    desc.includes('collectible')
  ) {
    return 'grind_hard';
  }

  if (
    t.secret === true &&
    (desc.includes('complete chapter') ||
      desc.includes('final mission') ||
      desc.includes('complete '))
  ) {
    return 'story_secret';
  }

  return 'action_misc';
}

function studentT4Cdf(x: number): number {
  const u = x / Math.sqrt(x * x + 4);
  return 0.5 + 0.25 * u * (3 - u * u);
}

/** Shared invariant helper: finite, clamped to [0, 100], rounded to 4 decimals, and non-decreasing. */
function finalizeQuantiles(raw: readonly number[]): number[] {
  const out = new Array<number>(LEVELS.length);
  for (let k = 0; k < LEVELS.length; k++) {
    const v = raw[k];
    let q = round4(clamp(typeof v === 'number' && Number.isFinite(v) ? v : 0.01, 0, 100));
    if (k > 0 && q < out[k - 1]) q = out[k - 1];
    out[k] = q;
  }
  return out;
}

function invertCdfQuantiles(cdf: (z: number) => number): number[] {
  const cdfAtFloor = cdf(Z_FLOOR);
  const qs = new Array<number>(LEVELS.length);

  for (let k = 0; k < LEVELS.length; k++) {
    const tau = LEVELS[k];
    if (cdfAtFloor >= tau) {
      qs[k] = 0.01;
      continue;
    }
    let lo = Z_FLOOR;
    let hi = Z_CEIL;
    for (let step = 0; step < BISECTION_STEPS; step++) {
      const mid = 0.5 * (lo + hi);
      if (cdf(mid) < tau) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    qs[k] = invLogitPct(0.5 * (lo + hi));
  }

  return finalizeQuantiles(qs);
}

/** Latest non-null/defined finite historical rate for trophy idx, defaulting to 0. */
function latestNonNullRate(states: UniqueState[], idx: number): number {
  for (let j = states.length - 1; j >= 0; j--) {
    const r = safeRate(states[j].rates[idx]);
    if (r !== null) return r;
  }
  return 0;
}

function applyPlatinumUpperBound(trophies: TrophyMeta[], rows: ForecastRow[]): void {
  const minNonPlat = [Infinity, Infinity, Infinity, Infinity, Infinity];
  let hasNonPlat = false;

  for (let i = 0; i < trophies.length; i++) {
    rows[i].q = finalizeQuantiles(rows[i].q);
    if ((trophies[i].tier ?? '').toLowerCase() !== 'platinum') {
      hasNonPlat = true;
      for (let k = 0; k < LEVELS.length; k++) {
        if (rows[i].q[k] < minNonPlat[k]) minNonPlat[k] = rows[i].q[k];
      }
    }
  }

  if (!hasNonPlat) return;

  for (let i = 0; i < trophies.length; i++) {
    if ((trophies[i].tier ?? '').toLowerCase() === 'platinum') {
      const capped = new Array<number>(LEVELS.length);
      for (let k = 0; k < LEVELS.length; k++) {
        capped[k] = Math.min(rows[i].q[k], minNonPlat[k]);
      }
      rows[i].q = finalizeQuantiles(capped);
    }
  }
}

function forecastCase(c: CaseFile): ForecastRow[] {
  const states = buildUniqueStates(c.history);
  const lastState = states[states.length - 1];
  const nLast = safePlayers(lastState.players);
  const horizonMinutes =
    typeof c.horizon_minutes === 'number' && Number.isFinite(c.horizon_minutes) && c.horizon_minutes > 0
      ? c.horizon_minutes
      : 0;
  const hDays = horizonMinutes / 1440;
  const sqrtHDays = Math.sqrt(hDays);

  // Filter out small launch-night QA states when at least 2 larger states exist.
  const minPlayers = Math.max(1500, Math.trunc(0.05 * nLast));
  const largeStates = states.filter((s) => s.players >= minPlayers);
  const fitStates = largeStates.length >= 2 ? largeStates : states;

  // Regime 1: Flat single-state regime (< 2 fit states).
  if (fitStates.length < 2) {
    const dilutionStrength = 1 / (1 + nLast / 1500);
    const hScale = Math.sqrt(Math.max(0.06, hDays * 16));
    const sig = Math.max(0.45, 1.38 * dilutionStrength * hScale + 0.35 * sqrtHDays);
    const shiftFactor = dilutionStrength * Math.min(1.8, hScale);

    const rows: ForecastRow[] = c.trophies.map((t, i) => {
      const r = latestNonNullRate(states, i);
      if (r >= 99.95) {
        return { id: t.id, q: finalizeQuantiles(PINNED_MAX_QUANTILES) };
      }
      const shift = (r > 0.02 ? -1.1 : -0.4) * shiftFactor;
      const baseL = logitPct(r);
      const raw = new Array<number>(LEVELS.length);
      for (let k = 0; k < LEVELS.length; k++) {
        raw[k] = invLogitPct(baseL + shift + Z_NORM[k] * sig);
      }
      return { id: t.id, q: finalizeQuantiles(raw) };
    });

    applyPlatinumUpperBound(c.trophies, rows);
    return rows;
  }

  // Regime 2: Multi-state decoupled N(t) & E_i(t) log-time OLS + split EB + 5-regime hurdle Student-t4.
  const m = fitStates.length;
  const uFirst = fitStates[0].u;
  const uLast = fitStates[m - 1].u;
  const targetMs = parseTargetMs(c, lastState.lastMs, horizonMinutes);
  const uTarget = toLogTime(targetMs);
  const du = Math.max(0.02, uTarget - uLast);

  // Reusable typed buffers & precomputed recency weights across fitStates.
  const xsBuf = new Float64Array(m);
  const ysBuf = new Float64Array(m);
  const wsBuf = new Float64Array(m);
  const recencyW = new Float64Array(m);

  for (let j = 0; j < m; j++) {
    const st = fitStates[j];
    const rw = Math.exp(-1.1 * Math.max(0, uLast - st.u));
    recencyW[j] = rw;
    xsBuf[j] = st.u;
    ysBuf[j] = Math.log(safePlayers(st.players));
    wsBuf[j] = j > 0 && st.players < fitStates[j - 1].players ? rw * 0.35 : rw;
  }

  const nOls = weightedOLS(xsBuf, ysBuf, wsBuf, m, 0.22);
  const nSlope = clamp(nOls.valid ? nOls.slope : 0.5, 0.02, 2.2);
  const nResid = nOls.resid;
  const dampN = 1 / (1 + 0.12 * du);
  const dampE = 1 / (1 + 0.08 * du);

  // Per-trophy raw E_i(t) fits and classifications.
  interface TrophyFitInfo {
    rNow: number;
    eNowRaw: number;
    eNowEst: number;
    category: Category;
    activeGrower: boolean;
    hasSlope: boolean;
    rawSlope: number;
    resid: number;
    infoWeight: number;
    kEff: number;
  }

  const fits: TrophyFitInfo[] = new Array(c.trophies.length);
  let globalSlopeNum = 0;
  let globalSlopeDen = 0;
  const groupNum = new Map<string, number>();
  const groupDen = new Map<string, number>();

  for (let i = 0; i < c.trophies.length; i++) {
    const t = c.trophies[i];
    const rNow = latestNonNullRate(states, i);
    const eNowRaw = Math.max(0, (nLast * rNow) / 100);
    const eNowEst = eNowRaw;
    const category = classifyTrophy(t);

    let eFirstRealEst = eNowEst;
    for (let j = 0; j < m; j++) {
      const rj = safeRate(fitStates[j].rates[i]);
      if (rj !== null) {
        eFirstRealEst = Math.max(0, (fitStates[j].players * rj) / 100);
        break;
      }
    }

    const activeGrower =
      rNow >= 0.03 ||
      (rNow > 0 && (eNowEst + 1) / Math.max(1, eFirstRealEst + 1) >= 1.75 && eNowEst >= 8);

    let hasSlope = false;
    let rawSlope = 0;
    let resid = 0.22;
    let infoWeight = 0;
    let validCount = 0;

    if (rNow > 0) {
      for (let j = 0; j < m; j++) {
        const rj = safeRate(fitStates[j].rates[i]);
        if (rj === null) continue;
        const eMid = Math.max(0, (fitStates[j].players * rj) / 100);
        xsBuf[validCount] = fitStates[j].u;
        ysBuf[validCount] = Math.log(Math.max(0.35, eMid));
        wsBuf[validCount] = recencyW[j] * Math.max(0.08, eMid / (eMid + 4));
        validCount++;
      }
      const eOls = weightedOLS(xsBuf, ysBuf, wsBuf, validCount, 0.22);
      resid = eOls.resid;
      if (eOls.valid) {
        hasSlope = true;
        rawSlope = clamp(eOls.slope, 0, 4.8);
        infoWeight = eOls.sumW * (eNowEst / (eNowEst + 8));

        const gKey = `${category}:${activeGrower ? 'active' : 'stagnant'}`;
        globalSlopeNum += infoWeight * rawSlope;
        globalSlopeDen += infoWeight;
        groupNum.set(gKey, (groupNum.get(gKey) ?? 0) + infoWeight * rawSlope);
        groupDen.set(gKey, (groupDen.get(gKey) ?? 0) + infoWeight);
      }
    } else {
      for (let j = 0; j < m; j++) {
        if (safeRate(fitStates[j].rates[i]) !== null) validCount++;
      }
    }

    fits[i] = {
      rNow,
      eNowRaw,
      eNowEst,
      category,
      activeGrower,
      hasSlope,
      rawSlope,
      resid,
      infoWeight,
      kEff: Math.max(1, validCount),
    };
  }

  const globalMeanSlope = globalSlopeDen > 0 ? globalSlopeNum / globalSlopeDen : nSlope;

  // Precompute all shared 5-scenario quantities once per case outside the trophy loop.
  const hMacro = Math.min(1, Math.sqrt(hDays / 4));
  const nUnc = Math.max(
    0.06,
    0.14 * sqrtHDays + 0.32 * nResid * Math.sqrt(Math.max(0.15, du / Math.max(0.05, uLast - uFirst))),
  );

  const ve = [
    1 - 0.42 * hMacro,
    1 - 0.18 * hMacro,
    1 - 0.04 * hMacro,
    1 - 0.02 * hMacro,
    1 + 0.16 * hMacro,
  ];
  const vn = [
    1 - 0.28 * hMacro,
    1 - 0.10 * hMacro,
    1,
    1 + 0.12 * hMacro,
    1 + 0.05 * hMacro,
  ];
  const dn = [-0.75 * nUnc, -0.30 * nUnc, 0, 0.90 * nUnc, -0.10 * nUnc];
  const de = [-0.22 * hMacro, -0.06 * hMacro, 0, 0.04 * hMacro, 0.26 * hMacro];

  const nScenarios = new Float64Array(NUM_SCENARIOS);
  const veDuDampE = new Float64Array(NUM_SCENARIOS);
  const zeroHurdleShift = new Float64Array(NUM_SCENARIOS);
  const duDampN = du * dampN;
  const duDampE = du * dampE;
  for (let s = 0; s < NUM_SCENARIOS; s++) {
    nScenarios[s] = Math.max(1, nLast * Math.exp(nSlope * vn[s] * duDampN + dn[s]));
    veDuDampE[s] = ve[s] * duDampE;
    zeroHurdleShift[s] = 0.6 * de[s] - 0.6 * dn[s] + 0.3 * Math.log(Math.max(0.2, ve[s]));
  }

  const baseSigCore = 0.18 + 0.18 * sqrtHDays;
  const breakHorizonScale = Math.min(1, hDays / 4.5);
  const stickyHorizonScale = Math.min(1, hDays / 2.5);
  const zSc = new Float64Array(NUM_SCENARIOS);

  // Forecast each trophy via hurdle mixture + 5-regime Student-t(df=4) CDF inversion.
  const rows: ForecastRow[] = new Array(c.trophies.length);

  for (let i = 0; i < c.trophies.length; i++) {
    const t = c.trophies[i];
    const f = fits[i];

    if (f.rNow >= 99.95) {
      rows[i] = { id: t.id, q: finalizeQuantiles(PINNED_MAX_QUANTILES) };
      continue;
    }

    // Empirical-Bayes shrunk slope.
    const gKey = `${f.category}:${f.activeGrower ? 'active' : 'stagnant'}`;
    const gNum = groupNum.get(gKey) ?? 0;
    const gDen = groupDen.get(gKey) ?? 0;
    const groupMean = (gNum + 1.0 * globalMeanSlope) / (gDen + 1.0);

    let slEb = f.hasSlope
      ? (f.infoWeight * f.rawSlope + TAU_EB * groupMean) / (f.infoWeight + TAU_EB)
      : groupMean;

    if (f.category === 'story_secret' && f.activeGrower && f.rNow <= 0.05) {
      slEb = Math.max(slEb, 1.45 * nSlope);
    }
    const slI = Math.max(0, Math.min(slEb, nSlope + 3.6 * Math.max(0.05, 1 - f.rNow / 100)));

    // Count & base uncertainty.
    const eLo = (Math.max(0, f.rNow - 0.005) / 100) * nLast;
    const eHi = (Math.min(100, f.rNow + 0.005) / 100) * nLast;
    const countSig = 0.5 * Math.log((eHi + 0.5) / Math.max(0.3, eLo + 0.5));
    const baseSig = baseSigCore * (1 + 0.12 * (1 - Math.min(1, (f.kEff - 1) / 5)));
    const trophySig = Math.sqrt(
      baseSig * baseSig + (0.32 * countSig) * (0.32 * countSig) + (0.28 * f.resid) * (0.28 * f.resid),
    );

    if (f.rNow <= 0.005) {
      // Zero hurdle mixture.
      const descLower = (t.description ?? '').toLowerCase();
      let pBase = 0.58;
      let zCondBase = logitPct(0.055);
      let condSig = 0.95;

      if (f.category === 'platinum') {
        pBase = 0.12;
        zCondBase = logitPct(0.022);
        condSig = 0.65;
      } else if (descLower.includes('100%')) {
        pBase = 0.40;
        zCondBase = logitPct(0.04);
        condSig = 0.85;
      }

      const pBreak = pBase * breakHorizonScale;
      const invCondSig = 1 / condSig;
      for (let s = 0; s < NUM_SCENARIOS; s++) {
        zSc[s] = zCondBase + zeroHurdleShift[s];
      }

      const cdf = (z: number): number => {
        let sum = 0;
        for (let s = 0; s < NUM_SCENARIOS; s++) {
          sum += SCENARIO_WEIGHTS[s] * studentT4Cdf((z - zSc[s]) * invCondSig);
        }
        return (1 - pBreak) + pBreak * sum;
      };

      rows[i] = { id: t.id, q: invertCdfQuantiles(cdf) };
      continue;
    }

    if (f.rNow <= 0.025 && !f.activeGrower) {
      // Stagnant ultra-rare sticky-floor mixture.
      const stickyFloorP =
        0.34 * Math.exp(-1.2 * Math.max(0, slI - 0.35 * nSlope)) * stickyHorizonScale;
      const invEffSig = 1 / Math.max(0.28, trophySig * 1.15);
      const minE = 0.8 * f.eNowRaw;
      for (let s = 0; s < NUM_SCENARIOS; s++) {
        const nSc = nScenarios[s];
        const eProj = f.eNowRaw * Math.exp(slI * veDuDampE[s] + de[s]);
        const eSc = Math.min(0.9999 * nSc, Math.max(minE, eProj));
        const pSc = clamp(eSc / nSc, 0.0001, 0.9999);
        zSc[s] = Math.log(pSc / (1 - pSc));
      }

      const cdf = (z: number): number => {
        let scSum = 0;
        for (let s = 0; s < NUM_SCENARIOS; s++) {
          scSum += SCENARIO_WEIGHTS[s] * studentT4Cdf((z - zSc[s]) * invEffSig);
        }
        return stickyFloorP * studentT4Cdf((z - Z_FLOOR) * 4) + (1 - stickyFloorP) * scSum;
      };

      rows[i] = { id: t.id, q: invertCdfQuantiles(cdf) };
      continue;
    }

    // Standard multi-scenario Student-t4 mixture.
    const invEffSig = 1 / (trophySig * (f.rNow <= 0.15 ? 1.16 : 1));
    const eAnchor = Math.max(1e-6, f.eNowRaw);
    for (let s = 0; s < NUM_SCENARIOS; s++) {
      const eProj = eAnchor * Math.exp(slI * veDuDampE[s] + de[s]);
      const pSc = clamp(eProj / nScenarios[s], 0.0001, 0.9999);
      zSc[s] = Math.log(pSc / (1 - pSc));
    }

    const cdf = (z: number): number => {
      let scSum = 0;
      for (let s = 0; s < NUM_SCENARIOS; s++) {
        scSum += SCENARIO_WEIGHTS[s] * studentT4Cdf((z - zSc[s]) * invEffSig);
      }
      return scSum;
    };

    rows[i] = { id: t.id, q: invertCdfQuantiles(cdf) };
  }

  applyPlatinumUpperBound(c.trophies, rows);
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();

for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(
      `${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`,
    );
  }
  if (!Array.isArray(c.history) || c.history.length === 0) {
    throw new Error(`${file} has no history`);
  }
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
