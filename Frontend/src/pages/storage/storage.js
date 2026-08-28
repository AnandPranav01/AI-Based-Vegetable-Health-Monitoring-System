import { requireAuth } from '/src/js/auth-guard.js';
import { injectSidebar } from '/src/js/navbar.js';

requireAuth();
injectSidebar('storage');

const STORAGE_DATA = [
  {
    emoji: '🍅', name: 'Tomato', location: 'counter',
    location_label: '🪴 Counter / Room Temp',
    temp: '18–21°C', humidity: 'Medium', life: '5–7 days',
    tips: [
      'Never refrigerate unripe tomatoes — cold destroys flavour and texture',
      'Store stem-side down to prevent moisture loss',
      'Keep away from direct sunlight',
      'Once cut, refrigerate in an airtight container for 2–3 days',
      'Store separately from cucumbers and leafy greens',
    ],
    doNot: 'Do NOT store in plastic bags — causes sweating and rot.',
  },
  {
    emoji: '🥕', name: 'Carrot', location: 'fridge',
    location_label: '🌡️ Refrigerator Crisper',
    temp: '0–4°C', humidity: 'High', life: '2–4 weeks',
    tips: [
      'Remove green tops before storing — they drain moisture from the root',
      'Store in a container with slightly damp paper towels',
      'Keep in the high-humidity crisper drawer',
      'Baby carrots can be stored in water in the fridge',
      'Keep away from apples and pears (ethylene gas)',
    ],
    doNot: 'Do NOT store near ethylene-producing fruits — causes bitterness.',
  },
  {
    emoji: '🥬', name: 'Spinach', location: 'fridge',
    location_label: '🌡️ Refrigerator',
    temp: '1–4°C', humidity: 'High', life: '3–5 days',
    tips: [
      'Do not wash until ready to use',
      'Store in a loose plastic bag with a dry paper towel',
      'Remove any damaged leaves immediately',
      'Keep in the coldest part of the fridge',
      'Use within 3 days for maximum nutrition',
    ],
    doNot: 'Do NOT freeze raw spinach — blanch first for up to 12 months frozen.',
  },
  {
    emoji: '🥔', name: 'Potato', location: 'pantry',
    location_label: '📦 Cool Dark Pantry',
    temp: '7–10°C', humidity: 'Medium', life: '2–3 weeks',
    tips: [
      'Store in a paper bag or mesh sack for airflow',
      'Keep in a cool, dark place away from light',
      'Do not refrigerate — cold converts starch to sugar',
      'Remove any sprouting potatoes quickly',
      'Store away from onions — they release gases that speed spoilage',
    ],
    doNot: 'Do NOT refrigerate — creates acrylamide (harmful compound) when cooked.',
  },
  {
    emoji: '🥦', name: 'Broccoli', location: 'fridge',
    location_label: '🌡️ Refrigerator Crisper',
    temp: '0–4°C', humidity: 'High', life: '3–5 days',
    tips: [
      'Store unwashed in an open plastic bag',
      'Broccoli can be misted with water to stay crisp',
      'Place upright in a glass of water like flowers for extra freshness',
      'Blanch and freeze for up to 12 months',
      'Use within 3 days for highest Vitamin C content',
    ],
    doNot: 'Do NOT seal in an airtight container — broccoli needs airflow.',
  },
  {
    emoji: '🥒', name: 'Cucumber', location: 'counter',
    location_label: '🪴 Counter / Fridge',
    temp: '10–15°C', humidity: 'Medium', life: '5–7 days',
    tips: [
      'Prefer room temperature over fridge (sensitive to cold)',
      'Wrap in a paper towel and a loose plastic bag if refrigerating',
      'Store away from tomatoes and melons',
      'Once cut, cover with plastic wrap and refrigerate',
      'Best consumed within 3 days of cutting',
    ],
    doNot: 'Do NOT store below 10°C — causes chilling injury and pitting.',
  },
  {
    emoji: '🌶️', name: 'Bell Pepper', location: 'fridge',
    location_label: '🌡️ Refrigerator Crisper',
    temp: '7–10°C', humidity: 'Medium', life: '1–2 weeks',
    tips: [
      'Store whole peppers unwashed in the crisper',
      'Red, yellow and orange last longer than green',
      'Slice and freeze in a single layer for up to 10 months',
      'Use ripened peppers within 3–4 days',
      'Keep dry — moisture causes mould',
    ],
    doNot: 'Do NOT store cut peppers at room temperature for more than 2 hours.',
  },
  {
    emoji: '🧅', name: 'Onion', location: 'pantry',
    location_label: '📦 Cool Dark Pantry',
    temp: '7–10°C', humidity: 'Low', life: '1–2 months',
    tips: [
      'Store in a mesh bag or open basket for airflow',
      'Keep in a cool, dark, dry place',
      'Never store near potatoes — both release gases',
      'Spring onions can be stood in water on the countertop',
      'Halved onions: wrap tightly and refrigerate (use within 2 weeks)',
    ],
    doNot: 'Do NOT store in plastic bags — traps moisture and causes rot.',
  },
  {
    emoji: '🧄', name: 'Garlic', location: 'pantry',
    location_label: '📦 Cool Dry Pantry',
    temp: '13–18°C', humidity: 'Low', life: '3–5 months',
    tips: [
      'Store in a mesh bag or terracotta pot with holes',
      'Keep whole bulbs at room temperature away from light',
      'Single cloves last 1–2 weeks at room temp',
      'Peeled cloves can be stored in olive oil in the fridge',
      'Freeze peeled cloves for up to 3 months',
    ],
    doNot: 'Do NOT refrigerate whole bulbs — causes sprouting and rubberiness.',
  },
  {
    emoji: '🌽', name: 'Corn', location: 'fridge',
    location_label: '🌡️ Refrigerator',
    temp: '0–4°C', humidity: 'High', life: '1–3 days',
    tips: [
      'Keep husks on until ready to eat — they protect freshness',
      'Refrigerate immediately after purchase',
      'Corn loses sweetness rapidly — eat within 1–2 days',
      'Blanch and freeze for up to 8 months',
      'Pre-shucked corn should be tightly wrapped in plastic',
    ],
    doNot: 'Do NOT leave at room temperature — sugars turn to starch within hours.',
  },
  {
    emoji: '🍆', name: 'Eggplant', location: 'counter',
    location_label: '🪴 Counter / Room Temp',
    temp: '10–13°C', humidity: 'Medium', life: '3–5 days',
    tips: [
      'Keep at room temperature away from direct sunlight',
      'Refrigeration causes texture deterioration',
      'Use within 3 days of purchase for best flavor',
      'Store away from ethylene-producing fruits',
      'Do not cut until ready to cook — oxidises quickly',
    ],
    doNot: 'Do NOT store below 10°C — cold injury causes brown flesh and bitterness.',
  },
  {
    emoji: '🥑', name: 'Avocado', location: 'counter',
    location_label: '🪴 Counter / Fridge',
    temp: 'Varies', humidity: 'Low', life: '2–7 days',
    tips: [
      'Keep unripe avocados at room temperature to ripen',
      'Speed up ripening by placing next to a banana',
      'Once ripe, refrigerate to slow further ripening (up to 5 days)',
      'Rub cut surfaces with lemon juice to prevent browning',
      'Store cut avocado with the pit to slow oxidation',
    ],
    doNot: 'Do NOT refrigerate unripe avocados — they will never ripen properly.',
  },
];

let activeFilter = 'all';
let searchQuery = '';

function render() {
  const grid = document.getElementById('storageGrid');
  const noEl = document.getElementById('noStorage');
  const filtered = STORAGE_DATA.filter(v => {
    const catMatch = activeFilter === 'all' || v.location === activeFilter;
    const searchMatch = !searchQuery || v.name.toLowerCase().includes(searchQuery);
    return catMatch && searchMatch;
  });

  if (filtered.length === 0) {
    grid.innerHTML = '';
    noEl.style.display = 'block';
    return;
  }
  noEl.style.display = 'none';

  grid.innerHTML = filtered.map(v => `
    <div class="storage-card">
      <div class="storage-veg">${v.emoji}</div>
      <div class="storage-veg-name">${v.name}</div>
      <div class="storage-method">${v.location_label}</div>
      <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.75rem;">
        <span class="badge badge-accent">🌡️ ${v.temp}</span>
        <span class="badge badge-success">💧 ${v.humidity} Humidity</span>
        <span class="badge badge-warning">⏱️ ${v.life}</span>
      </div>
      <ul class="storage-tips-list">
        ${v.tips.map(t => `<li>${t}</li>`).join('')}
      </ul>
      <div style="margin-top:0.75rem;padding:0.6rem 0.85rem;background:var(--danger-glow);border:1px solid rgba(255,71,87,0.2);border-radius:10px;font-size:0.78rem;color:var(--danger);">
        ⚠️ ${v.doNot}
      </div>
    </div>
  `).join('');
}

// Filter tabs
document.querySelectorAll('.tabs .tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tabs .tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeFilter = btn.dataset.filter;
    render();
  });
});

// Search
document.getElementById('searchInput').addEventListener('input', (e) => {
  searchQuery = e.target.value.toLowerCase().trim();
  render();
});

render();
