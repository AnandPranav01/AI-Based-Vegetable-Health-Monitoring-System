// Shared toast utility
export function showToast(message, type = 'success', duration = 3000) {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = `toast${type !== 'success' ? ' ' + type : ''}`;
  const icons = { success: '✅', error: '❌', warning: '⚠️' };
  toast.innerHTML = `<span>${icons[type] || '✅'}</span> ${message}`;
  document.body.appendChild(toast);
  setTimeout(() => toast.classList.add('show'), 50);
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 400);
  }, duration);
}

// Tab switching
export function initTabs(selector = '.tabs') {
  const tabsContainers = document.querySelectorAll(selector);
  tabsContainers.forEach(container => {
    container.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const targetId = btn.dataset.tab;
        container.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        const target = document.getElementById(targetId);
        if (target) target.classList.add('active');
      });
    });
  });
}

// Format date
export function formatDate(ts) {
  return new Date(ts).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}

// Animate number count up
export function countUp(el, target, duration = 1500) {
  let start = 0;
  const step = target / (duration / 16);
  const timer = setInterval(() => {
    start += step;
    if (start >= target) { el.textContent = target; clearInterval(timer); }
    else el.textContent = Math.round(start);
  }, 16);
}
