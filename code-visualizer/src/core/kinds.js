// Visual vocabulary shared by the layout (shape/size) and the renderers (colour/emoji).
// Palette follows Scratch's category colours (so kids transfer what they know) but with softer,
// modern tones. `ink` is the text colour that stays readable on the block colour.

export const KINDS = {
  start: { label: 'Start', color: '#FFBF00', ink: '#4A3500', emoji: '🏁', shape: 'hat' },
  end: { label: 'Done', color: '#8A94A6', ink: '#fff', emoji: '🏁', shape: 'pill' },
  decision: { label: 'Question', color: '#FFAB19', ink: '#4A2F00', emoji: '❓', shape: 'hex' },
  loop: { label: 'Repeat', color: '#FF8C1A', ink: '#4A2600', emoji: '🔁', shape: 'hex' },
  switch: { label: 'Choose', color: '#FFAB19', ink: '#4A2F00', emoji: '🔀', shape: 'hex' },
  return: { label: 'Finish', color: '#8A94A6', ink: '#fff', emoji: '↩️', shape: 'pill' },
  break: { label: 'Stop loop', color: '#FF6B6B', ink: '#fff', emoji: '🛑', shape: 'pill' },
  continue: { label: 'Next round', color: '#FFAB19', ink: '#4A2F00', emoji: '⏭️', shape: 'pill' },
  'pin-config': { label: 'Get ready', color: '#0FBD8C', ink: '#fff', emoji: '⚙️', shape: 'card' },
  output: { label: 'Output', color: '#9966FF', ink: '#fff', emoji: '💡', shape: 'card' },
  input: { label: 'Sense', color: '#2DB4D8', ink: '#fff', emoji: '👀', shape: 'card' },
  wait: { label: 'Wait', color: '#FFAB19', ink: '#4A2F00', emoji: '⏱️', shape: 'card' },
  serial: { label: 'Talk', color: '#CF63CF', ink: '#fff', emoji: '💬', shape: 'card' },
  bus: { label: 'Chip talk', color: '#0FBD8C', ink: '#fff', emoji: '🔌', shape: 'card' },
  call: { label: 'My block', color: '#FF6680', ink: '#fff', emoji: '🧩', shape: 'card' },
  trigger: { label: 'Interrupt', color: '#F5B700', ink: '#4A3500', emoji: '⚡', shape: 'card' },
  callback: { label: 'Callback', color: '#F5B700', ink: '#4A3500', emoji: '⚡', shape: 'card' },
  library: { label: 'Gadget', color: '#4C97FF', ink: '#fff', emoji: '🤖', shape: 'card' },
  declare: { label: 'Variable', color: '#FF8C1A', ink: '#4A2600', emoji: '📦', shape: 'card' },
  process: { label: 'Do', color: '#59C059', ink: '#fff', emoji: '✏️', shape: 'card' },
  globals: { label: 'Setup', color: '#6B7A99', ink: '#fff', emoji: '📋', shape: 'card' },
};

export const kindMeta = (kind) => KINDS[kind] || KINDS.process;

// Specific emoji by action category (falls back to the kind's emoji).
export const CATEGORY_EMOJI = {
  'PWM out': '🔆', Buzzer: '🔊', 'Analog in': '📈', 'Digital in': '👀', 'Pulse in': '📏', Servo: '🦾',
  'I²C': '🔌', SPI: '🔌', Variable: '📦', Init: '🔢', Step: '➕',
};
