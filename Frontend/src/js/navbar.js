import { getUser, logout } from '/src/js/auth-guard.js';

export function injectSidebar(activePage) {
  const user = getUser();
  const initial = user ? user.name.charAt(0).toUpperCase() : 'U';
  const name = user ? user.name : 'User';
  const email = user ? user.email : '';

  const navItems = [
    { icon: '🏠', label: 'Dashboard',       href: '/src/pages/dashboard/index.html',  id: 'dashboard' },
    { icon: '📷', label: 'Scan Vegetable',  href: '/src/pages/scan/index.html',        id: 'scan' },
    { icon: '📊', label: 'Analysis Results',href: '/src/pages/results/index.html',     id: 'results' },
    { icon: '💬', label: 'AI Chatbot',      href: '/src/pages/chatbot/index.html',     id: 'chatbot' },
    { icon: '🥗', label: 'Diet Planning',   href: '/src/pages/diet/index.html',        id: 'diet' },
    { icon: '👨‍🍳', label: 'Recipes',         href: '/src/pages/recipes/index.html',    id: 'recipes' },
    { icon: '📦', label: 'Storage Tips',    href: '/src/pages/storage/index.html',     id: 'storage' },
  ];

  const sidebarHTML = `
    <aside class="sidebar">
      <div class="sidebar-logo">
        <div class="logo-icon">🥦</div>
        <div class="logo-text">VegHealth AI<span>Smart Monitoring System</span></div>
      </div>
      <nav class="nav-section">
        <div class="nav-label">Main Menu</div>
        ${navItems.map(item => `
          <a href="${item.href}" class="nav-item ${activePage === item.id ? 'active' : ''}" id="nav-${item.id}">
            <span class="nav-icon">${item.icon}</span>
            <span>${item.label}</span>
          </a>
        `).join('')}
      </nav>
      <div class="sidebar-footer">
        <div class="user-card">
          <div class="user-avatar">${initial}</div>
          <div class="user-info">
            <div class="name">${name}</div>
            <div class="email">${email}</div>
          </div>
        </div>
        <button class="btn-logout" id="logoutBtn">🚪 Sign Out</button>
      </div>
    </aside>
  `;

  document.body.insertAdjacentHTML('afterbegin', sidebarHTML);
  document.getElementById('logoutBtn').addEventListener('click', logout);
}
