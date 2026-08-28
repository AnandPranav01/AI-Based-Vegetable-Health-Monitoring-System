import { requireAuth } from '/src/js/auth-guard.js';
import { injectSidebar } from '/src/js/navbar.js';

requireAuth();
injectSidebar('recipes');

const RECIPES = [
  {
    id: 1, emoji: '🥑', name: 'Spinach & Avocado Power Salad', cat: 'salad',
    time: '10 min', difficulty: 'Easy', cal: 180,
    veg: ['Spinach', 'Avocado', 'Cucumber', 'Tomato'],
    ingredients: ['100g baby spinach', '1 avocado, diced', '1 cucumber, sliced', '10 cherry tomatoes', '2 tbsp olive oil', '1 lemon, juiced', 'Salt & pepper', 'Pumpkin seeds'],
    steps: ['Wash and dry spinach leaves.', 'Dice avocado and slice cucumber.', 'Combine all vegetables in a large bowl.', 'Drizzle with olive oil and lemon juice.', 'Season with salt & pepper.', 'Top with pumpkin seeds and serve immediately.'],
    benefits: 'High in iron, healthy fats, Vitamin C and K. Excellent for immunity and energy.',
  },
  {
    id: 2, emoji: '🥦', name: 'Creamy Broccoli Soup', cat: 'soup',
    time: '25 min', difficulty: 'Easy', cal: 140,
    veg: ['Broccoli', 'Onion', 'Garlic', 'Carrot'],
    ingredients: ['400g broccoli florets', '1 onion, chopped', '3 garlic cloves', '1 carrot', '500ml vegetable stock', '100ml coconut cream', 'Salt & pepper', 'Fresh parsley'],
    steps: ['Sauté onion and garlic until soft.', 'Add carrot and broccoli, cook 5 min.', 'Pour in vegetable stock and simmer 15 min.', 'Blend until smooth.', 'Stir in coconut cream.', 'Season and serve with fresh parsley.'],
    benefits: 'Loaded with Vitamin C, K, and sulforaphane — a powerful cancer-fighting compound.',
  },
  {
    id: 3, emoji: '🥕', name: 'Rainbow Veggie Stir Fry', cat: 'stir-fry',
    time: '20 min', difficulty: 'Medium', cal: 195,
    veg: ['Carrot', 'Bell Pepper', 'Broccoli', 'Spinach'],
    ingredients: ['2 carrots, julienned', '1 red pepper, sliced', '100g broccoli', 'Handful spinach', '3 tbsp soy sauce', '1 tbsp sesame oil', '2 garlic cloves, minced', 'Ginger, sesame seeds'],
    steps: ['Heat sesame oil in wok on high heat.', 'Add garlic and ginger, fry 30 seconds.', 'Add carrots and peppers, stir fry 3 min.', 'Add broccoli and cook 2 more min.', 'Add spinach and soy sauce, toss well.', 'Serve over brown rice, topped with sesame seeds.'],
    benefits: 'Beta-carotene from carrots, Vitamin C from peppers, iron from spinach — a complete meal.',
  },
  {
    id: 4, emoji: '🍅', name: 'Tomato Garlic Shakshuka', cat: 'curry',
    time: '30 min', difficulty: 'Medium', cal: 210,
    veg: ['Tomato', 'Spinach', 'Bell Pepper', 'Onion'],
    ingredients: ['4 large tomatoes, diced', '1 cup spinach', '1 bell pepper, diced', '1 onion, sliced', '4 eggs', '2 tsp cumin', '1 tsp paprika', 'Fresh cilantro'],
    steps: ['Sauté onion and pepper in olive oil.', 'Add cumin and paprika, cook 1 min.', 'Add tomatoes and simmer 10 min.', 'Stir in spinach until wilted.', 'Make 4 wells and crack eggs in.', 'Cover and cook until eggs set. Garnish with cilantro.'],
    benefits: 'Lycopene from tomatoes, protein from eggs, iron from spinach — heart-healthy delight.',
  },
  {
    id: 5, emoji: '🥒', name: 'Cucumber Mint Raita', cat: 'snack',
    time: '10 min', difficulty: 'Easy', cal: 80,
    veg: ['Cucumber', 'Mint'],
    ingredients: ['2 cucumbers, grated', '200g Greek yogurt', 'Handful fresh mint', '1 garlic clove, minced', '1 tsp cumin', 'Salt', 'Lemon juice'],
    steps: ['Grate cucumber and squeeze out excess water.', 'Mix yogurt until smooth.', 'Combine cucumber with yogurt.', 'Add mint, garlic, cumin and salt.', 'Add a squeeze of lemon.', 'Chill for 30 min before serving.'],
    benefits: 'Probiotic-rich, hydrating and cooling. Excellent for digestion and gut health.',
  },
  {
    id: 6, emoji: '🥬', name: 'Green Immunity Smoothie', cat: 'smoothie',
    time: '5 min', difficulty: 'Easy', cal: 120,
    veg: ['Spinach', 'Cucumber', 'Celery'],
    ingredients: ['2 cups spinach', '1 cucumber', '2 celery stalks', '1 green apple', '1 lemon, juiced', '1 tsp ginger', '250ml cold water', 'Ice cubes'],
    steps: ['Wash all vegetables thoroughly.', 'Roughly chop cucumber and celery.', 'Add all ingredients to blender.', 'Blend on high for 60 seconds.', 'Add more water if too thick.', 'Serve immediately over ice.'],
    benefits: 'Chlorophyll-rich detox drink with Vitamin C. Boosts energy and supports liver health.',
  },
  {
    id: 7, emoji: '🥔', name: 'Spiced Roasted Potato Wedges', cat: 'snack',
    time: '40 min', difficulty: 'Easy', cal: 250,
    veg: ['Potato', 'Rosemary'],
    ingredients: ['4 large potatoes', '3 tbsp olive oil', '1 tsp smoked paprika', '1 tsp garlic powder', '1 tsp rosemary', 'Sea salt & pepper', 'Parsley'],
    steps: ['Preheat oven to 200°C.', 'Wash potatoes and cut into wedges.', 'Toss with olive oil and spices.', 'Arrange on baking tray skin-side down.', 'Roast 35–40 min until golden and crispy.', 'Garnish with fresh parsley and serve with dip.'],
    benefits: 'Good source of potassium and Vitamin C. Baked, not fried — a healthier comfort food.',
  },
  {
    id: 8, emoji: '🌶️', name: 'Stuffed Bell Pepper Bowls', cat: 'curry',
    time: '45 min', difficulty: 'Medium', cal: 280,
    veg: ['Bell Pepper', 'Tomato', 'Spinach', 'Onion'],
    ingredients: ['4 bell peppers', '1 cup brown rice', '1 onion, diced', '1 tomato, diced', 'Handful spinach', '1 tsp cumin', 'Cheese (optional)', 'Olive oil'],
    steps: ['Preheat oven to 190°C.', 'Cook rice according to package.', 'Sauté onion, add tomato and spinach.', 'Mix with cooked rice and cumin.', 'Cut tops off peppers and remove seeds.', 'Fill peppers with mixture. Bake 25 min.'],
    benefits: 'Highest Vitamin C content of any recipe. A complete, balanced meal in a natural bowl.',
  },
  {
    id: 9, emoji: '🥗', name: 'Carrot Ginger Immunity Soup', cat: 'soup',
    time: '35 min', difficulty: 'Easy', cal: 115,
    veg: ['Carrot', 'Ginger', 'Onion', 'Garlic'],
    ingredients: ['500g carrots, chopped', '1 large onion', '4 garlic cloves', '2 inch ginger', '700ml veg stock', 'Coconut milk', 'Turmeric', 'Coriander'],
    steps: ['Sauté onion, garlic and ginger.', 'Add turmeric and cook 1 min.', 'Add carrots and stock, simmer 20 min.', 'Blend until smooth.', 'Stir in coconut milk.', 'Serve with coriander and crusty bread.'],
    benefits: 'Anti-inflammatory ginger + beta-carotene carrots = powerful immunity boost. Great for cold & flu season.',
  },
];

let activeFilter = 'all';
let searchQuery = '';

function renderRecipes() {
  const grid = document.getElementById('recipeGrid');
  const noRec = document.getElementById('noRecipes');
  const filtered = RECIPES.filter(r => {
    const catMatch = activeFilter === 'all' || r.cat === activeFilter;
    const searchMatch = !searchQuery ||
      r.name.toLowerCase().includes(searchQuery) ||
      r.veg.some(v => v.toLowerCase().includes(searchQuery));
    return catMatch && searchMatch;
  });

  if (filtered.length === 0) {
    grid.innerHTML = '';
    noRec.style.display = 'block';
    return;
  }
  noRec.style.display = 'none';

  grid.innerHTML = filtered.map(r => `
    <div class="recipe-card" data-id="${r.id}">
      <div class="recipe-img" style="font-size:5rem;">${r.emoji}</div>
      <div class="recipe-body">
        <div class="recipe-title">${r.name}</div>
        <div class="recipe-meta">
          <span>⏱️ ${r.time}</span>
          <span>📶 ${r.difficulty}</span>
          <span>🔥 ${r.cal} kcal</span>
        </div>
        <div style="margin-top:0.75rem;display:flex;gap:0.35rem;flex-wrap:wrap;">
          ${r.veg.map(v => `<span class="badge badge-success" style="font-size:0.68rem;">${v}</span>`).join('')}
        </div>
        <button class="btn btn-primary btn-sm" style="margin-top:1rem;width:100%;" data-id="${r.id}">
          📖 View Recipe
        </button>
      </div>
    </div>
  `).join('');

  grid.querySelectorAll('[data-id]').forEach(el => {
    el.addEventListener('click', () => openModal(+el.dataset.id));
  });
}

function openModal(id) {
  const r = RECIPES.find(x => x.id === id);
  if (!r) return;
  document.getElementById('modalTitle').textContent = `${r.emoji} ${r.name}`;
  document.getElementById('modalBody').innerHTML = `
    <div style="display:flex;gap:1rem;margin-bottom:1.25rem;flex-wrap:wrap;">
      <span class="badge badge-success">⏱️ ${r.time}</span>
      <span class="badge badge-accent">📶 ${r.difficulty}</span>
      <span class="badge badge-warning">🔥 ${r.cal} kcal</span>
    </div>
    <div class="safety-banner safe" style="margin-bottom:1.25rem;">
      <div class="safety-banner-icon">💡</div>
      <div><div class="safety-banner-title">Health Benefits</div><div class="safety-banner-text">${r.benefits}</div></div>
    </div>
    <div class="grid-2" style="gap:1.5rem;">
      <div>
        <div style="font-weight:700;margin-bottom:0.75rem;font-size:0.95rem;">🛒 Ingredients</div>
        <ul style="list-style:none;display:flex;flex-direction:column;gap:0.4rem;">
          ${r.ingredients.map(i => `<li style="font-size:0.83rem;color:var(--text-secondary);display:flex;gap:0.5rem;"><span style="color:var(--primary);flex-shrink:0;">●</span>${i}</li>`).join('')}
        </ul>
      </div>
      <div>
        <div style="font-weight:700;margin-bottom:0.75rem;font-size:0.95rem;">📋 Method</div>
        <ol style="list-style:none;display:flex;flex-direction:column;gap:0.5rem;">
          ${r.steps.map((s, i) => `<li style="font-size:0.83rem;color:var(--text-secondary);display:flex;gap:0.6rem;"><span style="color:var(--primary);font-weight:700;flex-shrink:0;min-width:18px;">${i+1}.</span>${s}</li>`).join('')}
        </ol>
      </div>
    </div>
  `;
  document.getElementById('recipeModal').classList.add('open');
}

document.getElementById('modalClose').addEventListener('click', () =>
  document.getElementById('recipeModal').classList.remove('open'));
document.getElementById('recipeModal').addEventListener('click', (e) => {
  if (e.target === document.getElementById('recipeModal'))
    document.getElementById('recipeModal').classList.remove('open');
});

// Filter chips
document.getElementById('filterChips').querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    document.querySelectorAll('#filterChips .chip').forEach(c => {
      c.classList.remove('active');
      c.style.borderColor = '';
      c.style.color = '';
    });
    chip.classList.add('active');
    chip.style.borderColor = 'var(--primary)';
    chip.style.color = 'var(--primary)';
    activeFilter = chip.dataset.cat;
    renderRecipes();
  });
});

// Search
document.getElementById('searchInput').addEventListener('input', (e) => {
  searchQuery = e.target.value.toLowerCase().trim();
  renderRecipes();
});

renderRecipes();
