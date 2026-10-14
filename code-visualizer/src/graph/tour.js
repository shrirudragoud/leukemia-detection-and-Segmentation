// A "guided run" through the scene: the sequence of blocks the board would execute, so a learner can
// watch the code run. Questions alternate yes/no on successive visits, loops go round twice, switches
// cycle through their cases, and calls to the learner's own blocks are followed inline.
// (This is a walkthrough, not an interpreter: it never evaluates conditions.)

export function buildTour(scene, { lanes = null, maxSteps = 70 } = {}) {
  const out = new Map();
  for (const e of scene.edges) (out.get(e.from) ?? out.set(e.from, []).get(e.from)).push(e);
  const byId = new Map(scene.nodes.map((n) => [n.id, n]));
  const startOf = (laneId) => scene.nodes.find((n) => n.lane === laneId && n.kind === 'start');
  const wanted = lanes || scene.lanes.filter((l) => l.role === 'setup' || l.role === 'loop').map((l) => l.id);
  const steps = [];

  function walk(startNode, depth) {
    const visits = new Map();
    let node = startNode;
    let via = null;
    while (node && steps.length < maxSteps) {
      const seen = visits.get(node.id) ?? 0;
      if (seen >= 2 && node.kind !== 'loop') break; // a full second round has been shown
      visits.set(node.id, seen + 1);
      steps.push({ node, via });

      if (node.ref && depth === 0) {
        const lane = scene.lanes.find((l) => l.fn === node.ref);
        const s = lane && startOf(lane.id);
        if (s) walk(s, depth + 1);
      }

      const edges = out.get(node.id) || [];
      if (!edges.length) break;
      let pick = edges[0];
      if (edges.length > 1) {
        if (node.kind === 'switch') pick = edges[seen % edges.length];
        else {
          const wantYes = node.kind === 'loop' ? seen < 2 : seen % 2 === 0;
          pick = edges.find((e) => e.label === (wantYes ? 'yes' : 'no')) || edges.find((e) => e.kind !== 'back') || edges[0];
        }
      }
      via = pick;
      node = byId.get(pick.to);
    }
  }

  for (const id of wanted) {
    const s = startOf(id);
    if (s) walk(s, 0);
  }
  return steps;
}
