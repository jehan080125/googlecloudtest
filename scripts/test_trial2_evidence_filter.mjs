import assert from 'node:assert/strict';
import {
  DISASTER_EPITAPH_TRIAL2_EVIDENCE_ALLOWLIST,
  filterCollectedEvidenceForTrial,
  filterCourtInventoryForTrial,
  mergeCourtInventorySources,
  resolveInvestigationPhaseForTrialShortcut,
} from '../frontend/src/data/disasterEpitaphTrials.js';
import { mergeCourtEvidenceDisplay } from '../frontend/src/data/evidenceAssets.js';

const episode = {
  episode_id: 'disaster_epitaph',
  trial_exclude_evidence: {
    trial_epitaph_2: [
      'ev_ep_autopsy',
      'ev_ep_medical',
      'ev_ep_vx_info',
      'ev_ep_glasses',
      'ev_ep_cctv_club',
      'ev_ep_club_flyer',
      'ev_ep_doctor_opinion',
    ],
  },
};

const trial1Inventory = [
  'ev_ep_club_flyer',
  'ev_ep_glasses',
  'ev_ep_cctv_club',
  'ev_ep_autopsy',
  'ev_ep_vx_info',
  'ev_ep_medical',
  'ev_ep_doctor_opinion',
  'ev_ep_cctv_car',
  'ev_ep_autodrive_log',
  'ev_ep_wiretap',
  'ev_ep_anthony_id',
  'ev_ep_laptop',
  'ev_ep_server_log',
];

const filtered = filterCourtInventoryForTrial(trial1Inventory, episode, 'trial_epitaph_2');

assert.deepEqual(filtered, [
  'ev_ep_cctv_car',
  'ev_ep_autodrive_log',
  'ev_ep_wiretap',
  'ev_ep_anthony_id',
  'ev_ep_laptop',
  'ev_ep_server_log',
]);

assert.equal(filtered.length, 6);
assert.ok(!filtered.includes('ev_ep_club_flyer'));

const collected = filterCollectedEvidenceForTrial(
  [
    { id: 'ev_ep_club_flyer', courtEvidenceId: 'ev_ep_club_flyer', name: 'club' },
    { id: 'ev_ep_wiretap', courtEvidenceId: 'ev_ep_wiretap', name: 'wiretap' },
  ],
  episode,
  'trial_epitaph_2',
);
assert.equal(collected.length, 1);
assert.equal(collected[0].courtEvidenceId, 'ev_ep_wiretap');

assert.equal(resolveInvestigationPhaseForTrialShortcut(2), 'garage');
assert.equal(resolveInvestigationPhaseForTrialShortcut(1), 'initial');
assert.equal(DISASTER_EPITAPH_TRIAL2_EVIDENCE_ALLOWLIST.length, 8);

const merged = mergeCourtInventorySources(
  ['ev_ep_laptop', 'ev_ep_server_log'],
  [],
  [{ courtEvidenceId: 'ev_ep_wiretap', name: 'wiretap' }],
);
assert.deepEqual(
  filterCourtInventoryForTrial(merged, episode, 'trial_epitaph_2').sort(),
  ['ev_ep_laptop', 'ev_ep_server_log', 'ev_ep_wiretap'].sort(),
);

const display = mergeCourtEvidenceDisplay(
  [],
  [{ courtEvidenceId: 'ev_ep_wiretap', name: '도청기', desc: '녹음기' }],
  {
    episode_id: 'disaster_epitaph',
    evidences: [{ id: 'ev_ep_wiretap', name: '도청기', description: '녹음기' }],
  },
);
assert.equal(display.length, 1);
assert.equal(display[0].id, 'ev_ep_wiretap');
assert.equal(display[0].name, '도청기');

console.log('trial2 evidence filter tests passed');
