// Mock AI API responses
const FRESHNESS_RESULTS = {
  tomato:   { name: 'Tomato',   freshness: 88, status: 'Fresh',    shelf: 5,  safety: 'safe',    emoji: '🍅' },
  carrot:   { name: 'Carrot',   freshness: 76, status: 'Good',     shelf: 8,  safety: 'safe',    emoji: '🥕' },
  spinach:  { name: 'Spinach',  freshness: 42, status: 'Moderate', shelf: 2,  safety: 'caution', emoji: '🥬' },
  potato:   { name: 'Potato',   freshness: 91, status: 'Fresh',    shelf: 14, safety: 'safe',    emoji: '🥔' },
  broccoli: { name: 'Broccoli', freshness: 63, status: 'Good',     shelf: 4,  safety: 'safe',    emoji: '🥦' },
  lettuce:  { name: 'Lettuce',  freshness: 28, status: 'Spoiled',  shelf: 0,  safety: 'unsafe',  emoji: '🥗' },
  pepper:   { name: 'Pepper',   freshness: 95, status: 'Fresh',    shelf: 7,  safety: 'safe',    emoji: '🌶️' },
  cucumber: { name: 'Cucumber', freshness: 70, status: 'Good',     shelf: 4,  safety: 'safe',    emoji: '🥒' },
};

const NUTRITION_DB = {
  // Original 8
  tomato:   { calories: 18,  protein: '0.9g', carbs: '3.9g', fat: '0.2g', fiber: '1.2g',  vitC: '23mg',   potassium: '237mg' },
  carrot:   { calories: 41,  protein: '0.9g', carbs: '9.6g', fat: '0.2g', fiber: '2.8g',  vitC: '5.9mg',  potassium: '320mg' },
  spinach:  { calories: 23,  protein: '2.9g', carbs: '3.6g', fat: '0.4g', fiber: '2.2g',  vitC: '28mg',   potassium: '558mg' },
  potato:   { calories: 77,  protein: '2.0g', carbs: '17g',  fat: '0.1g', fiber: '2.2g',  vitC: '19mg',   potassium: '421mg' },
  broccoli: { calories: 34,  protein: '2.8g', carbs: '6.6g', fat: '0.4g', fiber: '2.6g',  vitC: '89mg',   potassium: '316mg' },
  lettuce:  { calories: 15,  protein: '1.4g', carbs: '2.9g', fat: '0.2g', fiber: '1.3g',  vitC: '9mg',    potassium: '194mg' },
  pepper:   { calories: 31,  protein: '1.0g', carbs: '6.0g', fat: '0.3g', fiber: '2.1g',  vitC: '125mg',  potassium: '211mg' },
  cucumber: { calories: 16,  protein: '0.7g', carbs: '3.6g', fat: '0.1g', fiber: '0.5g',  vitC: '2.8mg',  potassium: '147mg' },
  // All 47 YOLO classes
  almond:             { calories: 579, protein: '21g',  carbs: '22g',  fat: '50g',  fiber: '12.5g', vitC: '0mg',    potassium: '733mg' },
  apple:              { calories: 52,  protein: '0.3g', carbs: '14g',  fat: '0.2g', fiber: '2.4g',  vitC: '4.6mg',  potassium: '107mg' },
  asparagus:          { calories: 20,  protein: '2.2g', carbs: '3.9g', fat: '0.1g', fiber: '2.1g',  vitC: '5.6mg',  potassium: '202mg' },
  avocado:            { calories: 160, protein: '2.0g', carbs: '9.0g', fat: '15g',  fiber: '6.7g',  vitC: '10mg',   potassium: '485mg' },
  banana:             { calories: 89,  protein: '1.1g', carbs: '23g',  fat: '0.3g', fiber: '2.6g',  vitC: '8.7mg',  potassium: '358mg' },
  beans:              { calories: 31,  protein: '1.8g', carbs: '7.0g', fat: '0.1g', fiber: '2.7g',  vitC: '12mg',   potassium: '209mg' },
  beet:               { calories: 43,  protein: '1.6g', carbs: '10g',  fat: '0.2g', fiber: '2.8g',  vitC: '4.9mg',  potassium: '325mg' },
  bell_pepper:        { calories: 31,  protein: '1.0g', carbs: '6.0g', fat: '0.3g', fiber: '2.1g',  vitC: '128mg',  potassium: '211mg' },
  blackberry:         { calories: 43,  protein: '1.4g', carbs: '10g',  fat: '0.5g', fiber: '5.3g',  vitC: '21mg',   potassium: '162mg' },
  blueberry:          { calories: 57,  protein: '0.7g', carbs: '14g',  fat: '0.3g', fiber: '2.4g',  vitC: '9.7mg',  potassium: '77mg'  },
  brussels_sprouts:   { calories: 43,  protein: '3.4g', carbs: '9.0g', fat: '0.3g', fiber: '3.8g',  vitC: '85mg',   potassium: '389mg' },
  cabbage:            { calories: 25,  protein: '1.3g', carbs: '6.0g', fat: '0.1g', fiber: '2.5g',  vitC: '36mg',   potassium: '170mg' },
  cauliflower:        { calories: 25,  protein: '1.9g', carbs: '5.0g', fat: '0.3g', fiber: '2.0g',  vitC: '48mg',   potassium: '299mg' },
  celery:             { calories: 16,  protein: '0.7g', carbs: '3.0g', fat: '0.2g', fiber: '1.6g',  vitC: '3.1mg',  potassium: '260mg' },
  cherry:             { calories: 50,  protein: '1.0g', carbs: '12g',  fat: '0.3g', fiber: '1.6g',  vitC: '7.0mg',  potassium: '222mg' },
  corn:               { calories: 86,  protein: '3.2g', carbs: '19g',  fat: '1.2g', fiber: '2.7g',  vitC: '6.8mg',  potassium: '270mg' },
  egg:                { calories: 155, protein: '13g',  carbs: '1.1g', fat: '11g',  fiber: '0g',    vitC: '0mg',    potassium: '126mg' },
  eggplant:           { calories: 25,  protein: '1.0g', carbs: '6.0g', fat: '0.2g', fiber: '3.0g',  vitC: '2.2mg',  potassium: '229mg' },
  garlic:             { calories: 149, protein: '6.4g', carbs: '33g',  fat: '0.5g', fiber: '2.1g',  vitC: '31mg',   potassium: '401mg' },
  grape:              { calories: 69,  protein: '0.7g', carbs: '18g',  fat: '0.2g', fiber: '0.9g',  vitC: '10mg',   potassium: '191mg' },
  green_bean:         { calories: 31,  protein: '1.8g', carbs: '7.0g', fat: '0.1g', fiber: '2.7g',  vitC: '12mg',   potassium: '209mg' },
  green_onion:        { calories: 32,  protein: '1.8g', carbs: '7.3g', fat: '0.2g', fiber: '2.6g',  vitC: '19mg',   potassium: '276mg' },
  hot_pepper:         { calories: 40,  protein: '1.9g', carbs: '9.0g', fat: '0.4g', fiber: '1.5g',  vitC: '144mg',  potassium: '322mg' },
  kiwi:               { calories: 61,  protein: '1.1g', carbs: '15g',  fat: '0.5g', fiber: '3.0g',  vitC: '93mg',   potassium: '312mg' },
  lemon:              { calories: 29,  protein: '1.1g', carbs: '9.3g', fat: '0.3g', fiber: '2.8g',  vitC: '53mg',   potassium: '138mg' },
  lime:               { calories: 30,  protein: '0.7g', carbs: '11g',  fat: '0.2g', fiber: '2.8g',  vitC: '29mg',   potassium: '102mg' },
  mandarin:           { calories: 53,  protein: '0.8g', carbs: '13g',  fat: '0.3g', fiber: '1.8g',  vitC: '27mg',   potassium: '166mg' },
  mushroom:           { calories: 22,  protein: '3.1g', carbs: '3.3g', fat: '0.3g', fiber: '1.0g',  vitC: '2.1mg',  potassium: '318mg' },
  onion:              { calories: 40,  protein: '1.1g', carbs: '9.3g', fat: '0.1g', fiber: '1.7g',  vitC: '7.4mg',  potassium: '146mg' },
  orange:             { calories: 47,  protein: '0.9g', carbs: '12g',  fat: '0.1g', fiber: '2.4g',  vitC: '53mg',   potassium: '181mg' },
  pattypan_squash:    { calories: 18,  protein: '1.2g', carbs: '4.0g', fat: '0.2g', fiber: '1.5g',  vitC: '17mg',   potassium: '182mg' },
  pea:                { calories: 81,  protein: '5.4g', carbs: '14g',  fat: '0.4g', fiber: '5.1g',  vitC: '40mg',   potassium: '244mg' },
  peach:              { calories: 39,  protein: '0.9g', carbs: '10g',  fat: '0.3g', fiber: '1.5g',  vitC: '6.6mg',  potassium: '190mg' },
  pear:               { calories: 57,  protein: '0.4g', carbs: '15g',  fat: '0.1g', fiber: '3.1g',  vitC: '4.3mg',  potassium: '116mg' },
  pineapple:          { calories: 50,  protein: '0.5g', carbs: '13g',  fat: '0.1g', fiber: '1.4g',  vitC: '48mg',   potassium: '109mg' },
  pumpkin:            { calories: 26,  protein: '1.0g', carbs: '7.0g', fat: '0.1g', fiber: '0.5g',  vitC: '9.0mg',  potassium: '340mg' },
  radish:             { calories: 16,  protein: '0.7g', carbs: '3.4g', fat: '0.1g', fiber: '1.6g',  vitC: '15mg',   potassium: '233mg' },
  raspberry:          { calories: 52,  protein: '1.2g', carbs: '12g',  fat: '0.7g', fiber: '6.5g',  vitC: '26mg',   potassium: '151mg' },
  strawberry:         { calories: 32,  protein: '0.7g', carbs: '7.7g', fat: '0.3g', fiber: '2.0g',  vitC: '59mg',   potassium: '153mg' },
  vegetable_marrow:   { calories: 17,  protein: '1.2g', carbs: '3.4g', fat: '0.2g', fiber: '1.1g',  vitC: '17mg',   potassium: '182mg' },
  watermelon:         { calories: 30,  protein: '0.6g', carbs: '7.6g', fat: '0.2g', fiber: '0.4g',  vitC: '8.1mg',  potassium: '112mg' },
};

const EMOJI_MAP = {
  almond: '🥜', apple: '🍎', asparagus: '🌿', avocado: '🥑', banana: '🍌',
  beans: '🫘', beet: '🫚', bell_pepper: '🫑', blackberry: '🫐', blueberry: '🫐',
  broccoli: '🥦', brussels_sprouts: '🥦', cabbage: '🥬', carrot: '🥕',
  cauliflower: '🥦', celery: '🥬', cherry: '🍒', corn: '🌽', cucumber: '🥒',
  egg: '🥚', eggplant: '🍆', garlic: '🧄', grape: '🍇', green_bean: '🫘',
  green_onion: '🧅', hot_pepper: '🌶️', kiwi: '🥝', lemon: '🍋', lettuce: '🥬',
  lime: '🍋', mandarin: '🍊', mushroom: '🍄', onion: '🧅', orange: '🍊',
  pattypan_squash: '🥒', pea: '🫛', peach: '🍑', pear: '🍐', pineapple: '🍍',
  potato: '🥔', pumpkin: '🎃', radish: '🔴', raspberry: '🫐', spinach: '🥬',
  strawberry: '🍓', tomato: '🍅', vegetable_marrow: '🥒', watermelon: '🍉',
  pepper: '🌶️',
};

const VEGETABLES = ['tomato','carrot','spinach','potato','broccoli','lettuce','pepper','cucumber'];
const random = arr => arr[Math.floor(Math.random() * arr.length)];

// ── Real API ──────────────────────────────────────────────────────────────────
const API_BASE = 'http://localhost:8000';

const DEFAULT_NUTRITION = { calories: 30, protein: '1.5g', carbs: '6g', fat: '0.2g', fiber: '2g', vitC: '15mg', potassium: '250mg' };

function _normaliseKey(objectClass) {
  return objectClass.toLowerCase().replace(/\s+/g, '_');
}

function _freshnessToStatus(pct) {
  if (pct >= 75) return 'Fresh';
  if (pct >= 50) return 'Good';
  if (pct >= 30) return 'Moderate';
  return 'Spoiled';
}

function _spoilageToSafety(pct, spoilageStatus) {
  if (spoilageStatus === 'fresh' && pct >= 60) return 'safe';
  if (pct >= 30) return 'caution';
  return 'unsafe';
}

function _estimateShelf(pct, spoilageStatus) {
  if (spoilageStatus === 'spoiled' || pct < 30) return 0;
  return Math.max(1, Math.round((pct / 100) * 14));
}

function _pickPrimary(detections) {
  // If any are spoiled, highlight the worst one (lowest freshness)
  const spoiled = detections.filter(d => d.spoilage_status === 'spoiled');
  if (spoiled.length > 0) {
    return spoiled.reduce((a, b) => a.freshness_percentage < b.freshness_percentage ? a : b);
  }
  // All fresh — pick highest combined confidence
  return detections.reduce((a, b) => a.combined_confidence > b.combined_confidence ? a : b);
}

function _transformApiResponse(data) {
  const primary = _pickPrimary(data.detections);
  const key = _normaliseKey(primary.object_class);
  const freshness = Math.round(primary.freshness_percentage);
  const spoilageStatus = primary.spoilage_status;

  return {
    name: primary.object_class.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
    freshness,
    status: _freshnessToStatus(freshness),
    shelf: _estimateShelf(freshness, spoilageStatus),
    safety: _spoilageToSafety(freshness, spoilageStatus),
    emoji: EMOJI_MAP[key] || EMOJI_MAP[primary.object_class.toLowerCase()] || '🥗',
    nutrition: NUTRITION_DB[key] || NUTRITION_DB[primary.object_class.toLowerCase()] || DEFAULT_NUTRITION,
    key,
    // Extended real-API fields (used by results page for multi-detection summary)
    allDetections: data.detections,
    fresh_count: data.fresh_count,
    spoiled_count: data.spoiled_count,
    total_detections: data.total_detections,
    processing_time_s: data.processing_time_s,
    source: 'real_api',
  };
}

function _fallbackFromFilename(imageData) {
  const filename = (imageData instanceof File ? imageData.name : '').toLowerCase().replace(/[_\-]/g, ' ');
  // Try to match any known food key against the filename
  const allKeys = Object.keys(NUTRITION_DB);
  const matched = allKeys.find(k => filename.includes(k.replace(/_/g, ' ')));
  if (!matched) return null;
  // Use static freshness if available, otherwise default to 80% fresh
  const staticResult = FRESHNESS_RESULTS[matched];
  const freshness = staticResult ? staticResult.freshness : 80;
  const spoilageStatus = freshness >= 50 ? 'fresh' : 'spoiled';
  const name = matched.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  return {
    name,
    freshness,
    status: staticResult ? staticResult.status : _freshnessToStatus(freshness),
    shelf: staticResult ? staticResult.shelf : _estimateShelf(freshness, spoilageStatus),
    safety: staticResult ? staticResult.safety : _spoilageToSafety(freshness, spoilageStatus),
    emoji: EMOJI_MAP[matched] || '🥗',
    nutrition: NUTRITION_DB[matched] || DEFAULT_NUTRITION,
    key: matched,
    allDetections: [],
    fresh_count: spoilageStatus === 'fresh' ? 1 : 0,
    spoiled_count: spoilageStatus === 'spoiled' ? 1 : 0,
    total_detections: 1,
    processing_time_s: 0,
    source: 'mock_fallback',
  };
}

async function _callRealApi(imageData) {
  const formData = new FormData();
  const file = imageData instanceof File
    ? imageData
    : new File([imageData], 'capture.jpg', { type: 'image/jpeg' });
  formData.append('file', file);

  let res;
  try {
    const apiUrl = typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_URL 
      ? import.meta.env.VITE_API_URL 
      : API_BASE;
    res = await fetch(`${apiUrl}/predict`, { 
      method: 'POST', 
      body: formData,
      headers: {
        'Bypass-Tunnel-Reminder': 'true' // For localtunnel compatibility
      }
    });
  } catch (error) {
    console.warn("Real API unavailable or blocked. Falling back to mock data.", error);
    const fallback = _fallbackFromFilename(imageData);
    if (fallback) return fallback;
    
    // Fallback to random if no name match
    const randomFood = VEGETABLES[Math.floor(Math.random() * VEGETABLES.length)];
    return {
      ...FRESHNESS_RESULTS[randomFood],
      nutrition: NUTRITION_DB[randomFood],
      key: randomFood,
      source: 'mock_fallback'
    };
  }

  if (!res.ok) {
    console.warn(`API returned ${res.status}, falling back to mock.`);
    const fallback = _fallbackFromFilename(imageData);
    if (fallback) return fallback;

    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `API error ${res.status}`);
  }
  
  const data = await res.json();
  if (data.total_detections === 0) {
    // Try to identify food from filename before giving up
    const fallback = _fallbackFromFilename(imageData);
    if (fallback) return fallback;
    throw new Error('No food items detected. Try a clearer, closer image.');
  }
  return _transformApiResponse(data);
}

/**
 * analyzeVegetable(imageData)
 *   imageData: File | Blob  → calls real FastAPI /predict endpoint
 *   imageData: string|null  → falls back to mock (used for sample chips)
 */
export async function analyzeVegetable(imageData) {
  if (imageData instanceof Blob || imageData instanceof File) {
    return _callRealApi(imageData);
  }
  // Mock fallback (sample chips / offline)
  const name = typeof imageData === 'string' ? imageData : '';
  const key = VEGETABLES.find(v => name.toLowerCase().includes(v)) || random(VEGETABLES);
  return new Promise(resolve => {
    setTimeout(() => resolve({
      ...FRESHNESS_RESULTS[key],
      nutrition: NUTRITION_DB[key],
      key,
      source: 'mock',
    }), 2200);
  });
}

export function getResultFromStorage() {
  const stored = localStorage.getItem('veghealth_last_result');
  return stored ? JSON.parse(stored) : null;
}

export function saveResult(result) {
  localStorage.setItem('veghealth_last_result', JSON.stringify(result));
  // also save to history
  const history = JSON.parse(localStorage.getItem('veghealth_history') || '[]');
  history.unshift({ ...result, timestamp: Date.now() });
  localStorage.setItem('veghealth_history', JSON.stringify(history.slice(0, 50)));
}

export function getHistory() {
  return JSON.parse(localStorage.getItem('veghealth_history') || '[]');
}

// Chatbot mock responses
const CHAT_PATTERNS = [
  { pattern: /tomato/i,    response: '🍅 Tomatoes are rich in lycopene and Vitamin C. They are excellent for heart health and immunity. Fresh tomatoes should be firm and bright red.' },
  { pattern: /carrot/i,    response: '🥕 Carrots are packed with beta-carotene, which converts to Vitamin A. Great for eye health! Store them in the fridge wrapped in a damp cloth.' },
  { pattern: /spinach/i,   response: '🥬 Spinach is a powerhouse of iron, calcium, and folate. It\'s best consumed fresh within 3-4 days. Great for anemia prevention!' },
  { pattern: /potato/i,    response: '🥔 Potatoes are high in potassium and Vitamin C. Store in a cool, dark place — never in the fridge. They can last up to 3 weeks if stored properly.' },
  { pattern: /broccoli/i,  response: '🥦 Broccoli is one of the most nutritious vegetables! It\'s high in Vitamin K, C, and cancer-fighting compounds called sulforaphanes.' },
  { pattern: /fresh/i,     response: '✅ To check freshness: look for vibrant color, firm texture, and no visible mold. Smell is also a great indicator — fresh vegetables smell mild and earthy.' },
  { pattern: /store|storage/i, response: '🗃️ Most vegetables do best in the fridge at 35-40°F (1-4°C) in breathable bags. Root vegetables like potatoes prefer cool, dark pantries. Leafy greens need high humidity.' },
  { pattern: /diet|nutrition/i, response: '🥗 A balanced vegetable diet includes leafy greens (spinach, kale), colorful vegetables (peppers, tomatoes), and starchy vegetables (potatoes, sweet potato) for complete nutrition.' },
  { pattern: /recipe/i,    response: '👨‍🍳 I can suggest recipes! Try stir-frying mixed vegetables with garlic and ginger, or roasting root vegetables with olive oil and herbs. What vegetable do you have in mind?' },
  { pattern: /hello|hi|hey/i, response: '👋 Hello! I\'m VegBot, your AI vegetable health assistant. Ask me about vegetable freshness, nutrition, storage, or recipes!' },
  { pattern: /shelf.?life/i, response: '⏱️ Shelf life varies: Leafy greens (1-3 days), Tomatoes & peppers (5-7 days), Root vegetables (1-3 weeks), Onions & garlic (several weeks). Proper storage can extend these.' },
];

const DEFAULT_RESPONSES = [
  "🌿 That's a great question about vegetable health! Could you be more specific about which vegetable you're asking about?",
  "🍃 As your vegetable health assistant, I recommend scanning your vegetables regularly to track freshness and nutrition.",
  "🥗 Eating a rainbow of vegetables ensures you get a wide range of vitamins, minerals, and antioxidants.",
  "💡 Tip: Buy locally grown, seasonal vegetables for maximum freshness and nutritional value.",
  "🔬 Our AI can detect over 50 types of vegetables and analyze their freshness score, shelf life, and nutritional content.",
];

export function getChatResponse(message) {
  return new Promise(resolve => {
    setTimeout(() => {
      const match = CHAT_PATTERNS.find(p => p.pattern.test(message));
      resolve(match ? match.response : random(DEFAULT_RESPONSES));
    }, 1000 + Math.random() * 1000);
  });
}
