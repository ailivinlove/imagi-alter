/* eslint-disable @typescript-eslint/no-use-before-define */
/* eslint-disable @typescript-eslint/no-explicit-any */

type Trial = { text: string };

type GenerateArgs = { k: number; seed?: number; numTrials?: number };

const DEFAULT_TRIAL_COUNT = 60;
const MAX_TOKENS = 7;

const ANGLES = [15, 30, 36, 45, 54, 60, 72, 75, 90, 105, 108, 120, 135, 144, 150, 165, 180];
const ANGLE_BONUS = new Set([36, 54, 72, 108, 144]);

const WORD_TO_NUMBER: Record<string, number> = {
  one: 1,
  two: 2,
  three: 3,
  four: 4,
  five: 5
};

const DIRECTIONS = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"] as const;
type Direction = typeof DIRECTIONS[number];

const EPS = 1e-6;

type Vec = { x: number; y: number };

type Constraint =
  | { kind: "rotate"; target: string; angle: number; center: string; source: string }
  | { kind: "lattice"; target: string; base: string; dx: number; dy: number; raw: LatticeText }
  | { kind: "project"; target: string; source: string; lineA: string; lineB: string }
  | { kind: "orientation"; center: string; a: string; b: string; direction: "counterclockwise" | "clockwise" }
  | { kind: "orthant"; target: string; base: string; direction: Direction }
  | { kind: "collinear"; target: string; a: string; b: string }
  | { kind: "coincident"; a: string; b: string }
  | { kind: "unknown" };

type LatticeText = {
  wordX: keyof typeof WORD_TO_NUMBER;
  dirX: "east" | "west";
  wordY: keyof typeof WORD_TO_NUMBER;
  dirY: "north" | "south";
};

interface FamilyContext {
  contextTrials: Trial[];
  matchTrial: Trial;
  family: "orientation" | "orthant" | "collinearity";
}

interface GeneratorState {
  rng: () => number;
  graph: ConstraintGraph;
  windowTexts: string[];
  windowSet: Set<string>;
  k: number;
  angleUsage: Set<number>;
  lastFamily: FamilyContext["family"] | null;
  letters: LetterPool;
}

class LetterPool {
  private readonly used = new Set<string>();
  private readonly rng: () => number;

  constructor(rng: () => number) {
    this.rng = rng;
  }

  next(preferExisting?: boolean): string {
    const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
    if (preferExisting) {
      const existing = alphabet.filter((c) => this.used.has(c));
      if (existing.length > 0) {
        return existing[Math.floor(this.rng() * existing.length)];
      }
    }
    const choices = alphabet.filter((c) => !this.used.has(c));
    if (choices.length === 0) {
      const existing = alphabet[Math.floor(this.rng() * alphabet.length)];
      this.used.add(existing);
      return existing;
    }
    const pick = choices[Math.floor(this.rng() * choices.length)];
    this.used.add(pick);
    return pick;
  }

  mark(letter: string) {
    this.used.add(letter);
  }
}

class ConstraintGraph {
  private readonly rng: () => number;
  private readonly points = new Map<string, Vec>();

  constructor(rng: () => number) {
    this.rng = rng;
  }

  apply(constraint: Constraint) {
    switch (constraint.kind) {
      case "rotate":
        this.applyRotate(constraint);
        break;
      case "lattice":
        this.applyLattice(constraint);
        break;
      case "project":
        this.applyProject(constraint);
        break;
      case "orientation":
      case "orthant":
      case "collinear":
      case "coincident":
      case "unknown":
        // no structural update needed
        break;
      default:
        exhaustive(constraint);
    }
  }

  private ensurePoint(letter: string): Vec {
    if (!this.points.has(letter)) {
      const theta = this.rng() * Math.PI * 2;
      const radius = 5 + this.rng() * 5;
      this.points.set(letter, { x: radius * Math.cos(theta), y: radius * Math.sin(theta) });
    }
    return this.points.get(letter)!;
  }

  private setPoint(letter: string, value: Vec) {
    if (this.points.has(letter)) {
      const existing = this.points.get(letter)!;
      if (!closeVec(existing, value)) {
        this.points.set(letter, {
          x: (existing.x + value.x) / 2,
          y: (existing.y + value.y) / 2
        });
      }
    } else {
      this.points.set(letter, value);
    }
  }

  private applyRotate({ target, angle, center, source }: Extract<Constraint, { kind: "rotate" }>) {
    const centerPoint = this.ensurePoint(center);
    const sourcePoint = this.ensurePoint(source);
    const rad = (angle * Math.PI) / 180;
    const dx = sourcePoint.x - centerPoint.x;
    const dy = sourcePoint.y - centerPoint.y;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);
    const rotated: Vec = {
      x: centerPoint.x + dx * cos - dy * sin,
      y: centerPoint.y + dx * sin + dy * cos
    };
    this.setPoint(target, rotated);
  }

  private applyLattice({ target, base, dx, dy }: Extract<Constraint, { kind: "lattice" }>) {
    const basePoint = this.ensurePoint(base);
    this.setPoint(target, { x: basePoint.x + dx, y: basePoint.y + dy });
  }

  private applyProject({ target, source, lineA, lineB }: Extract<Constraint, { kind: "project" }>) {
    const a = this.ensurePoint(lineA);
    const b = this.ensurePoint(lineB);
    const s = this.ensurePoint(source);
    const ab = { x: b.x - a.x, y: b.y - a.y };
    const abLenSq = ab.x * ab.x + ab.y * ab.y;
    let projected: Vec;
    if (abLenSq < EPS) {
      projected = { ...a };
    } else {
      const as = { x: s.x - a.x, y: s.y - a.y };
      const t = (as.x * ab.x + as.y * ab.y) / abLenSq;
      projected = { x: a.x + ab.x * t, y: a.y + ab.y * t };
    }
    this.setPoint(target, projected);
  }

  getPoint(letter: string): Vec | undefined {
    return this.points.get(letter);
  }
}

function closeVec(a: Vec, b: Vec, eps = EPS) {
  return Math.abs(a.x - b.x) <= eps && Math.abs(a.y - b.y) <= eps;
}

function createRng(seed = Date.now()): () => number {
  let x = seed >>> 0;
  return () => {
    x += 0x6d2b79f5;
    let t = x;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function generateTrials(args: GenerateArgs): Trial[] {
  const k = Math.max(1, Math.floor(args.k || 1));
  const rng = createRng(args.seed);
  const graph = new ConstraintGraph(rng);
  const letters = new LetterPool(rng);
  const state: GeneratorState = {
    rng,
    graph,
    windowTexts: [],
    windowSet: new Set(),
    k,
    angleUsage: new Set(),
    lastFamily: null,
    letters
  };
  const numTrials = Math.max(k + 5, Math.floor(args.numTrials || DEFAULT_TRIAL_COUNT));
  const trials: Trial[] = [];

  // Seed window with filler to avoid early matches
  for (let i = 0; i < k; i++) {
    const filler = createFiller(state);
    appendTrial(state, trials, filler);
  }

  const families: FamilyContext["family"][] = ["orientation", "orthant", "collinearity"];
  let familyIndex = 0;
  while (trials.length < numTrials) {
    const family = families[familyIndex % families.length];
    familyIndex += 1;
    if (family === state.lastFamily) {
      continue;
    }
    const seq = createFamilySequence(state, family);
    appendSequence(state, trials, seq);
    if (trials.length >= numTrials) break;
    const fillerCount = Math.max(1, Math.floor(state.rng() * (k + 1)));
    for (let i = 0; i < fillerCount && trials.length < numTrials; i++) {
      const filler = createFiller(state);
      appendTrial(state, trials, filler);
    }
  }

  return trials.slice(0, numTrials);
}

function appendSequence(state: GeneratorState, trials: Trial[], seq: FamilyContext) {
  for (const trial of seq.contextTrials) {
    appendTrial(state, trials, trial);
  }
  appendTrial(state, trials, seq.matchTrial);
  state.lastFamily = seq.family;
}

function appendTrial(state: GeneratorState, trials: Trial[], trial: Trial) {
  const sanitized = sanitize(trial.text);
  enforceTokenLimit(sanitized);
  if (state.windowSet.has(sanitized)) {
    throw new Error(`Duplicate premise within window: ${sanitized}`);
  }
  trials.push({ text: sanitized });
  state.windowTexts.push(sanitized);
  state.windowSet.add(sanitized);
  if (state.windowTexts.length > state.k) {
    const removed = state.windowTexts.shift();
    if (removed) {
      state.windowSet.delete(removed);
    }
  }
  const constraint = parseConstraint(sanitized);
  state.graph.apply(constraint);
}

function createFamilySequence(state: GeneratorState, family: FamilyContext["family"]): FamilyContext {
  switch (family) {
    case "orientation":
      return createOrientationSequence(state);
    case "orthant":
      return createOrthantSequence(state);
    case "collinearity":
      return createCollinearitySequence(state);
    default:
      return exhaustive(family);
  }
}

function createOrientationSequence(state: GeneratorState): FamilyContext {
  const context: Trial[] = [];
  const slots = Math.max(1, state.k);
  for (let i = 0; i < slots - 1; i++) {
    context.push(createFiller(state));
  }
  const center = state.letters.next();
  const source = state.letters.next(true);
  const target = state.letters.next();
  const angle = weightedAngle(state);
  state.angleUsage.add(angle);
  const rotateText = `${target} rotate${angle} ${center} from ${source}`;
  context.push({ text: rotateText });
  const matchText = `${center} ${source} ${target} counterclockwise`;
  return {
    contextTrials: context,
    matchTrial: { text: matchText },
    family: "orientation"
  };
}

function createOrthantSequence(state: GeneratorState): FamilyContext {
  const context: Trial[] = [];
  const slots = Math.max(1, state.k);
  for (let i = 0; i < slots - 1; i++) {
    context.push(createFiller(state));
  }
  const base = state.letters.next();
  const target = state.letters.next();
  const dxWord = randomWord(state.rng);
  const dyWord = randomWord(state.rng);
  const dxDir = state.rng() < 0.5 ? "east" : "west";
  const dyDir = state.rng() < 0.5 ? "north" : "south";
  const dx = WORD_TO_NUMBER[dxWord] * (dxDir === "east" ? 1 : -1);
  const dy = WORD_TO_NUMBER[dyWord] * (dyDir === "north" ? 1 : -1);
  const latticeText = `${target} ${dxWord} ${dxDir} ${dyWord} ${dyDir} ${base}`;
  context.push({ text: latticeText });
  const direction = orthantFromVector(dx, dy);
  const matchText = `${target} ${direction} of ${base}`;
  return {
    contextTrials: context,
    matchTrial: { text: matchText },
    family: "orthant"
  };
}

function createCollinearitySequence(state: GeneratorState): FamilyContext {
  const context: Trial[] = [];
  const slots = Math.max(1, state.k);
  for (let i = 0; i < slots - 1; i++) {
    context.push(createFiller(state));
  }
  const target = state.letters.next();
  const source = state.letters.next(true);
  const lineA = state.letters.next();
  const lineB = state.letters.next();
  const projectText = `${target} project ${source} to line ${lineA},${lineB}`;
  context.push({ text: projectText });
  const matchText = `${target} same line as ${lineA},${lineB}`;
  return {
    contextTrials: context,
    matchTrial: { text: matchText },
    family: "collinearity"
  };
}

function createFiller(state: GeneratorState): Trial {
  if (state.rng() < 0.5) {
    const center = state.letters.next(true);
    const source = state.letters.next(true);
    const target = state.letters.next();
    const angle = ANGLES[Math.floor(state.rng() * ANGLES.length)];
    const text = `${target} rotate${angle} ${center} from ${source}`;
    return { text };
  }
  const base = state.letters.next(true);
  const target = state.letters.next();
  const dxWord = randomWord(state.rng);
  const dyWord = randomWord(state.rng);
  const dxDir = state.rng() < 0.5 ? "east" : "west";
  const dyDir = state.rng() < 0.5 ? "north" : "south";
  const text = `${target} ${dxWord} ${dxDir} ${dyWord} ${dyDir} ${base}`;
  return { text };
}

function randomWord(rng: () => number): keyof typeof WORD_TO_NUMBER {
  const words = Object.keys(WORD_TO_NUMBER) as (keyof typeof WORD_TO_NUMBER)[];
  return words[Math.floor(rng() * words.length)];
}

function weightedAngle(state: GeneratorState): number {
  const allowed = ANGLES.filter((angle) => angle !== 180);
  const weights = allowed.map((angle) => (ANGLE_BONUS.has(angle) ? 3 : 1));
  const total = weights.reduce((sum, w) => sum + w, 0);
  let pick = state.rng() * total;
  for (let i = 0; i < allowed.length; i++) {
    pick -= weights[i];
    if (pick <= 0) {
      return allowed[i];
    }
  }
  return allowed[allowed.length - 1];
}

function enforceTokenLimit(text: string) {
  const tokens = text.split(/\s+/);
  if (tokens.length > MAX_TOKENS) {
    throw new Error(`Premise exceeds ${MAX_TOKENS} tokens: ${text}`);
  }
}

function sanitize(text: string): string {
  let sanitized = text.replace(/opposite/gi, "inverted");
  sanitized = sanitized.replace(/\babout\b/gi, "").replace(/\bexactly\b/gi, "").replace(/\s+/g, " ").trim();
  return sanitized;
}

function orthantFromVector(dx: number, dy: number): Direction {
  const horizontal = dx > 0 ? "east" : "west";
  const vertical = dy > 0 ? "north" : "south";
  if (Math.abs(dx) < EPS) {
    return vertical as Direction;
  }
  if (Math.abs(dy) < EPS) {
    return horizontal as Direction;
  }
  return `${vertical}${horizontal === "east" ? "east" : "west"}` as Direction;
}

function parseConstraint(text: string): Constraint {
  const rotateMatch = text.match(/^([A-Z]) rotate(\d{2,3}) ([A-Z]) from ([A-Z])$/);
  if (rotateMatch) {
    return {
      kind: "rotate",
      target: rotateMatch[1],
      angle: parseInt(rotateMatch[2], 10),
      center: rotateMatch[3],
      source: rotateMatch[4]
    };
  }
  const latticeMatch = text.match(/^([A-Z]) (one|two|three|four|five) (east|west) (one|two|three|four|five) (north|south) ([A-Z])$/);
  if (latticeMatch) {
    const wordX = latticeMatch[2] as keyof typeof WORD_TO_NUMBER;
    const wordY = latticeMatch[4] as keyof typeof WORD_TO_NUMBER;
    const dirX = latticeMatch[3] as "east" | "west";
    const dirY = latticeMatch[5] as "north" | "south";
    const dx = WORD_TO_NUMBER[wordX] * (dirX === "east" ? 1 : -1);
    const dy = WORD_TO_NUMBER[wordY] * (dirY === "north" ? 1 : -1);
    return {
      kind: "lattice",
      target: latticeMatch[1],
      base: latticeMatch[6],
      dx,
      dy,
      raw: { wordX, dirX, wordY, dirY }
    };
  }
  const projectMatch = text.match(/^([A-Z]) project ([A-Z]) to line ([A-Z]),([A-Z])$/);
  if (projectMatch) {
    return {
      kind: "project",
      target: projectMatch[1],
      source: projectMatch[2],
      lineA: projectMatch[3],
      lineB: projectMatch[4]
    };
  }
  const orientationMatch = text.match(/^([A-Z]) ([A-Z]) ([A-Z]) (counterclockwise|clockwise)$/);
  if (orientationMatch) {
    return {
      kind: "orientation",
      center: orientationMatch[1],
      a: orientationMatch[2],
      b: orientationMatch[3],
      direction: orientationMatch[4] as "counterclockwise" | "clockwise"
    };
  }
  const orthantMatch = text.match(/^([A-Z]) (north|south|east|west|northeast|northwest|southeast|southwest) of ([A-Z])$/);
  if (orthantMatch) {
    return {
      kind: "orthant",
      target: orthantMatch[1],
      direction: orthantMatch[2] as Direction,
      base: orthantMatch[3]
    };
  }
  const collinearMatch = text.match(/^([A-Z]) same line as ([A-Z]),([A-Z])$/);
  if (collinearMatch) {
    return {
      kind: "collinear",
      target: collinearMatch[1],
      a: collinearMatch[2],
      b: collinearMatch[3]
    };
  }
  const coincidentMatch = text.match(/^([A-Z]) same position as ([A-Z])$/);
  if (coincidentMatch) {
    return {
      kind: "coincident",
      a: coincidentMatch[1],
      b: coincidentMatch[2]
    };
  }
  return { kind: "unknown" };
}

export function validateDerivedMatch(trials: Trial[], t: number, k: number): boolean {
  if (t <= 0 || t >= trials.length) return false;
  if (t <= k) return false;
  const windowStart = Math.max(0, t - k);
  const window = trials.slice(windowStart, t);
  const targetText = sanitize(trials[t].text);
  if (window.some((trial) => sanitize(trial.text) === targetText)) {
    return false;
  }
  const constraints = window.map((trial) => parseConstraint(sanitize(trial.text)));
  const graph = new ConstraintGraph(createRng(123));
  const knowledge = new Knowledge(graph);
  for (const constraint of constraints) {
    knowledge.apply(constraint);
  }
  const predicate = parseConstraint(targetText);
  return knowledge.isEntailed(predicate);
}

class Knowledge {
  private readonly graph: ConstraintGraph;
  private readonly constraints: Constraint[] = [];

  constructor(graph: ConstraintGraph) {
    this.graph = graph;
  }

  apply(constraint: Constraint) {
    this.constraints.push(constraint);
    this.graph.apply(constraint);
  }

  isEntailed(predicate: Constraint): boolean {
    switch (predicate.kind) {
      case "orientation":
        return this.entailsOrientation(predicate);
      case "orthant":
        return this.entailsOrthant(predicate);
      case "collinear":
        return this.entailsCollinearity(predicate);
      case "coincident":
        return this.entailsCoincidence(predicate);
      default:
        return false;
    }
  }

  private entailsOrientation(predicate: Extract<Constraint, { kind: "orientation" }>): boolean {
    const pointCenter = this.graph.getPoint(predicate.center);
    const pointA = this.graph.getPoint(predicate.a);
    const pointB = this.graph.getPoint(predicate.b);
    if (!pointCenter || !pointA || !pointB) {
      return false;
    }
    const orient = orientation(pointCenter, pointA, pointB);
    if (Math.abs(orient) < EPS) return false;
    return predicate.direction === "counterclockwise" ? orient > 0 : orient < 0;
  }

  private entailsOrthant(predicate: Extract<Constraint, { kind: "orthant" }>): boolean {
    const target = this.graph.getPoint(predicate.target);
    const base = this.graph.getPoint(predicate.base);
    if (!target || !base) return false;
    const dx = target.x - base.x;
    const dy = target.y - base.y;
    const derived = orthantFromVector(dx, dy);
    return derived === predicate.direction;
  }

  private entailsCollinearity(predicate: Extract<Constraint, { kind: "collinear" }>): boolean {
    const target = this.graph.getPoint(predicate.target);
    const a = this.graph.getPoint(predicate.a);
    const b = this.graph.getPoint(predicate.b);
    if (!target || !a || !b) return false;
    return Math.abs(orientation(a, b, target)) < 1e-5;
  }

  private entailsCoincidence(predicate: Extract<Constraint, { kind: "coincident" }>): boolean {
    const a = this.graph.getPoint(predicate.a);
    const b = this.graph.getPoint(predicate.b);
    if (!a || !b) return false;
    return closeVec(a, b, 1e-4);
  }
}

function orientation(center: Vec, a: Vec, b: Vec): number {
  const ax = a.x - center.x;
  const ay = a.y - center.y;
  const bx = b.x - center.x;
  const by = b.y - center.y;
  return ax * by - ay * bx;
}

function exhaustive(value: never): never {
  throw new Error(`Unhandled case: ${String(value)}`);
}

// Minimal self-check to exercise generator variety.
export function __selfCheck(seed = 12345) {
  const angles = new Set<number>();
  const trials = generateTrials({ k: 2, seed });
  trials.forEach((trial) => {
    const match = trial.text.match(/rotate(\d{2,3})/);
    if (match) {
      angles.add(parseInt(match[1], 10));
    }
  });
  return angles.size;
}

export type { Trial, GenerateArgs };
