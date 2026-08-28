import { requireAuth } from '/src/js/auth-guard.js';
import { injectSidebar } from '/src/js/navbar.js';

requireAuth();
injectSidebar('diet');

const GOALS = {
  weightloss: {
    icon: '⚖️', title: 'Weight Loss Plan',
    text: 'Low-calorie, high-fiber vegetables to keep you full while reducing caloric intake.',
    vegs: [
      { e: '🥬', n: 'Spinach',   b: 'Very low calorie (23 kcal), high iron & folate. Perfect base for salads.' },
      { e: '🥦', n: 'Broccoli',  b: 'High fiber, Vitamin C & K. Keeps you satiated longer.' },
      { e: '🥒', n: 'Cucumber',  b: 'Only 16 kcal per 100g. Excellent for hydration and weight control.' },
    ],
    plan: [
      ['Monday',    ['🥬 Spinach smoothie','🥗 Cucumber salad','🥦 Steamed broccoli','🥕 Carrot sticks']],
      ['Tuesday',   ['🥦 Broccoli soup','🥒 Cucumber wrap','🥬 Sautéed spinach','🍅 Cherry tomatoes']],
      ['Wednesday', ['🍅 Tomato juice','🥦 Broccoli stir fry','🥒 Cucumber raita','🥬 Spinach salad']],
      ['Thursday',  ['🥕 Carrot juice','🥗 Mixed greens','🌽 Stir-fried vegs','🥦 Broccoli florets']],
      ['Friday',    ['🥬 Green smoothie','🥒 Cucumber soup','🍅 Tomato salad','🥕 Sliced carrots']],
      ['Saturday',  ['🥦 Broccoli frittata','🥬 Spinach rice','🥒 Tzatziki bowl','🍅 Bruschetta']],
      ['Sunday',    ['🥕 Carrot oats','🥗 Buddha bowl','🥦 Roasted broccoli','🥬 Detox salad']],
    ],
  },
  immunity: {
    icon: '🛡️', title: 'Immunity Boost Plan',
    text: 'Vitamin C-rich and antioxidant-packed vegetables to strengthen your immune system.',
    vegs: [
      { e: '🌶️', n: 'Bell Pepper', b: 'Highest Vitamin C of any vegetable (125mg per 100g).' },
      { e: '🥦', n: 'Broccoli',    b: 'Rich in Vitamin C, E & beta-carotene for immune defense.' },
      { e: '🧄', n: 'Garlic',      b: 'Allicin compound has powerful antimicrobial properties.' },
    ],
    plan: [
      ['Monday',    ['🌶️ Pepper omelet','🥦 Immunity soup','🌶️ Stuffed pepper','🧄 Garlic broth']],
      ['Tuesday',   ['🥦 Broccoli juice','🌶️ Pepper salad','🥦 Garlic broccoli','🍅 Tomato soup']],
      ['Wednesday', ['🧄 Garlic toast','🌶️ Roasted peppers','🥦 Stir fry','🥕 Carrot sticks']],
      ['Thursday',  ['🥦 Smoothie','🧄 Garlic soup','🌶️ Fajita bowl','🥦 Broccoli bites']],
      ['Friday',    ['🌶️ Juice','🥦 Mixed salad','🧄 Roasted garlic','🌶️ Pepper hummus']],
      ['Saturday',  ['🥦 Pancakes','🌶️ Stuffed vegs','🧄 Pasta','🍅 Salsa bowl']],
      ['Sunday',    ['🌶️ Shakshuka','🥦 Buddha bowl','🧄 Soup','🌶️ Sliced peppers']],
    ],
  },
  muscle: {
    icon: '💪', title: 'Muscle Gain Plan',
    text: 'High-protein plant foods and energy-dense vegetables to fuel muscle growth.',
    vegs: [
      { e: '🥬', n: 'Spinach',  b: 'High plant protein (2.9g/100g) + iron for oxygen delivery to muscles.' },
      { e: '🥦', n: 'Broccoli', b: 'Contains sulforaphane which reduces muscle inflammation post-workout.' },
      { e: '🥔', n: 'Potato',   b: 'Complex carbs for sustained energy + potassium for muscle recovery.' },
    ],
    plan: [
      ['Monday',    ['🥔 Potato hash','🥬 Spinach pasta','🥦 Protein bowl','🥬 Protein shake']],
      ['Tuesday',   ['🥬 Green eggs','🥔 Jacket potato','🥦 Stir fry','🥔 Sweet potato']],
      ['Wednesday', ['🥦 Broccoli rice','🥬 Spinach dal','🥔 Potato curry','🥦 Roasted cuts']],
      ['Thursday',  ['🥔 Hash browns','🥦 Broccoli soup','🥬 Spinach wrap','🥔 Baked potato']],
      ['Friday',    ['🥬 Smoothie bowl','🥔 Potato salad','🥦 Stir fry','🥬 Protein salad']],
      ['Saturday',  ['🥦 Frittata','🥔 Stuffed potato','🥬 Pasta','🥦 Snack bites']],
      ['Sunday',    ['🥔 Pancakes','🥬 Rice bowl','🥦 Curry','🥔 Wedges']],
    ],
  },
  diabetes: {
    icon: '💉', title: 'Diabetes Control Plan',
    text: 'Low glycemic index vegetables that help stabilize blood sugar levels.',
    vegs: [
      { e: '🥬', n: 'Leafy Greens', b: 'Very low GI, high magnesium which is linked to lower diabetes risk.' },
      { e: '🥦', n: 'Broccoli',     b: 'Sulforaphane reduces blood glucose and improves insulin sensitivity.' },
      { e: '🥒', n: 'Cucumber',     b: 'Low carb (3.6g), high water content helps regulate blood sugar.' },
    ],
    plan: [
      ['Monday',    ['🥬 Green salad','🥦 Broccoli soup','🥒 Raita','🥬 Steamed greens']],
      ['Tuesday',   ['🥒 Cucumber juice','🥬 Salad wrap','🥦 Stir fry','🥒 Sliced snack']],
      ['Wednesday', ['🥦 Smoothie','🥒 Soup','🥬 Spinach rice','🥦 Florets']],
      ['Thursday',  ['🥬 Juice','🥦 Salad','🥒 Tzatziki','🥬 Sauté']],
      ['Friday',    ['🥒 Detox water','🥬 Bowl','🥦 Curry','🥒 Snack']],
      ['Saturday',  ['🥦 Egg cups','🥒 Salad','🥬 Pasta','🥦 Bites']],
      ['Sunday',    ['🥬 Smoothie','🥦 Bowl','🥒 Soup','🥬 Salad']],
    ],
  },
  heart: {
    icon: '❤️', title: 'Heart Health Plan',
    text: 'Potassium-rich and cholesterol-lowering vegetables for cardiovascular health.',
    vegs: [
      { e: '🍅', n: 'Tomato',  b: 'Lycopene reduces LDL cholesterol and blood pressure.' },
      { e: '🥬', n: 'Spinach', b: 'Nitrates help relax blood vessels and reduce blood pressure.' },
      { e: '🧅', n: 'Onion',   b: 'Quercetin reduces inflammation and improves heart health.' },
    ],
    plan: [
      ['Monday',    ['🍅 Juice','🥬 Salad','🧅 Soup','🍅 Snack']],
      ['Tuesday',   ['🥬 Smoothie','🍅 Pasta','🧅 Stir fry','🥬 Salad']],
      ['Wednesday', ['🍅 Shakshuka','🥬 Rice','🧅 Curry','🍅 Bites']],
      ['Thursday',  ['🧅 Toast','🍅 Salad','🥬 Bowl','🧅 Soup']],
      ['Friday',    ['🥬 Juice','🧅 Omelette','🍅 Bruschetta','🥬 Sauté']],
      ['Saturday',  ['🍅 Soup','🥬 Pasta','🧅 Roasted','🍅 Salsa']],
      ['Sunday',    ['🧅 Hash','🍅 Bowl','🥬 Curry','🧅 Snack']],
    ],
  },
  detox: {
    icon: '🌿', title: 'Detox & Cleanse Plan',
    text: 'Antioxidant and chlorophyll-rich vegetables to flush toxins and rejuvenate.',
    vegs: [
      { e: '🥬', n: 'Spinach',  b: 'Chlorophyll binds to toxins helping remove them from the body.' },
      { e: '🥕', n: 'Carrot',   b: 'Beta-carotene supports liver function and detox processes.' },
      { e: '🧄', n: 'Garlic',   b: 'Sulfur compounds activate liver enzymes to flush out toxins.' },
    ],
    plan: [
      ['Monday',    ['🥬 Detox juice','🥕 Soup','🧄 Garlic broth','🥬 Salad']],
      ['Tuesday',   ['🥕 Smoothie','🥬 Greens wrap','🧄 Pasta','🥕 Sticks']],
      ['Wednesday', ['🥬 Green juice','🧄 Soup','🥕 Salad','🥬 Sauté']],
      ['Thursday',  ['🧄 Toast','🥬 Bowl','🥕 Stir fry','🧄 Broth']],
      ['Friday',    ['🥕 Juice','🥬 Detox salad','🧄 Curry','🥕 Snack']],
      ['Saturday',  ['🥬 Smoothie','🥕 Soup','🧄 Roasted','🥬 Salad']],
      ['Sunday',    ['🧄 Oat bowl','🥬 Rice','🥕 Roast','🧄 Detox tea']],
    ],
  },
  energy: {
    icon: '⚡', title: 'Energy Boost Plan',
    text: 'Complex carb and iron-rich vegetables for sustained energy throughout the day.',
    vegs: [
      { e: '🥔', n: 'Potato',  b: 'Complex carbs provide long-lasting energy without sugar spikes.' },
      { e: '🥕', n: 'Carrot',  b: 'Natural sugars and B vitamins support energy metabolism.' },
      { e: '🌽', n: 'Corn',    b: 'High in thiamine (B1) which is essential for energy production.' },
    ],
    plan: [
      ['Monday',    ['🥔 Hash','🌽 Salad','🥕 Stir fry','🥔 Wedges']],
      ['Tuesday',   ['🥕 Juice','🥔 Soup','🌽 Rice','🥕 Snack']],
      ['Wednesday', ['🌽 Porridge','🥕 Pasta','🥔 Bake','🌽 Popcorn']],
      ['Thursday',  ['🥔 Pancakes','🌽 Bowl','🥕 Curry','🥔 Chips']],
      ['Friday',    ['🥕 Smoothie','🥔 Jacket','🌽 Chaat','🥕 Sticks']],
      ['Saturday',  ['🌽 Fritters','🥔 Salad','🥕 Soup','🌽 Snack']],
      ['Sunday',    ['🥔 Waffles','🥕 Bowl','🌽 Stir fry','🥔 Mash']],
    ],
  },
  balanced: {
    icon: '🥦', title: 'Balanced Diet Plan',
    text: 'A complete mix of vegetables covering all vitamins, minerals, and macronutrients.',
    vegs: [
      { e: '🍅', n: 'Tomato',  b: 'Vitamin C & lycopene for immunity and heart health.' },
      { e: '🥦', n: 'Broccoli', b: 'The most nutrient-dense vegetable — comprehensive vitamins.' },
      { e: '🥕', n: 'Carrot',  b: 'Beta-carotene for eye health and immune support.' },
    ],
    plan: [
      ['Monday',    ['🥦 Smoothie','🍅 Salad','🥕 Stir fry','🥦 Bites']],
      ['Tuesday',   ['🍅 Juice','🥦 Pasta','🥕 Soup','🍅 Snack']],
      ['Wednesday', ['🥕 Porridge','🥦 Bowl','🍅 Curry','🥕 Sticks']],
      ['Thursday',  ['🥦 Toast','🍅 Bruschetta','🥕 Rice','🥦 Chips']],
      ['Friday',    ['🍅 Shakshuka','🥕 Salad','🥦 Stir fry','🍅 Bites']],
      ['Saturday',  ['🥕 Pancakes','🥦 Soup','🍅 Wrap','🥕 Juice']],
      ['Sunday',    ['🥦 Frittata','🍅 Bowl','🥕 Curry','🥦 Roast']],
    ],
  },
};

let currentGoal = 'weightloss';

function render(goal) {
  const g = GOALS[goal];
  // Banner
  document.getElementById('goalBannerIcon').textContent = g.icon;
  document.getElementById('goalBannerTitle').textContent = g.title;
  document.getElementById('goalBannerText').textContent = g.text;

  // Diet table
  const tbody = document.getElementById('dietBody');
  tbody.innerHTML = g.plan.map(([day, meals]) => `
    <tr>
      <td><strong style="color:var(--primary)">${day}</strong></td>
      ${meals.map(m => `<td>${m}</td>`).join('')}
    </tr>
  `).join('');

  // Vegetable highlights
  document.getElementById('veggieHighlights').innerHTML = g.vegs.map(v => `
    <div class="card" style="text-align:center;cursor:default;">
      <div style="font-size:2.5rem;margin-bottom:0.5rem;">${v.e}</div>
      <div style="font-weight:700;font-size:1rem;margin-bottom:0.5rem;">${v.n}</div>
      <div style="font-size:0.78rem;color:var(--text-secondary);line-height:1.5;">${v.b}</div>
      <div style="margin-top:0.75rem;"><a href="/src/pages/storage/index.html" class="badge badge-success">📦 Storage Tips</a></div>
    </div>
  `).join('');
}

// Goal card selection
document.getElementById('goalGrid').querySelectorAll('.goal-card').forEach(card => {
  card.addEventListener('click', () => {
    document.querySelectorAll('.goal-card').forEach(c => c.classList.remove('selected'));
    card.classList.add('selected');
    currentGoal = card.dataset.goal;
    render(currentGoal);
  });
});

document.getElementById('generateBtn').addEventListener('click', () => render(currentGoal));

render(currentGoal);
