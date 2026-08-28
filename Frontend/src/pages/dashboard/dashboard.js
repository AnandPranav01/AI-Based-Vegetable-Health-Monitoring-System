import { requireAuth } from '/src/js/auth-guard.js';
import { injectSidebar } from '/src/js/navbar.js';
import { getHistory } from '/src/js/mock-api.js';
import { countUp, formatDate } from '/src/js/utils.js';

const user = requireAuth();
if (!user) throw new Error('unauthenticated');
injectSidebar('dashboard');

// Greeting
const hour = new Date().getHours();
const greet = hour < 12 ? 'Good Morning' : hour < 17 ? 'Good Afternoon' : 'Good Evening';
document.getElementById('greetingTitle').textContent = `${greet}, ${user.name.split(' ')[0]} 👋`;

// Load history
const history = getHistory();
const total = history.length;
const fresh = history.filter(h => h.freshness >= 71).length;
const mod = history.filter(h => h.freshness >= 41 && h.freshness < 71).length;
const spoiled = history.filter(h => h.freshness < 41).length;
const healthyPct = total ? Math.round((fresh / total) * 100) : 78;
const alertCount = total ? mod : 2;
const spoiledCount = total ? spoiled : 1;
const totalCount = total || 24;

// Animate stat cards
setTimeout(() => {
  countUp(document.getElementById('totalScans'), totalCount);
  countUp(document.getElementById('alertCount'), alertCount);
  countUp(document.getElementById('spoiledCount'), spoiledCount);
}, 200);
document.getElementById('healthyPct').innerHTML = `${healthyPct}<span style="font-size:1.2rem">%</span>`;

// Freshness bars
const fp = total ? Math.round((fresh / total) * 100) : 65;
const mp = total ? Math.round((mod / total) * 100) : 25;
const sp = total ? Math.round((spoiled / total) * 100) : 10;
setTimeout(() => {
  document.getElementById('pctFresh').textContent = `${fp}%`;
  document.getElementById('pctMod').textContent = `${mp}%`;
  document.getElementById('pctSpoil').textContent = `${sp}%`;
  document.getElementById('barFresh').style.width = fp + '%';
  document.getElementById('barMod').style.width = mp + '%';
  document.getElementById('barSpoil').style.width = sp + '%';
}, 400);

// History table
const tbody = document.getElementById('historyTableBody');
const emptyDiv = document.getElementById('emptyHistory');

const STATUS_COLOR = { Fresh: 'success', Good: 'success', Moderate: 'warning', Spoiled: 'danger' };

if (history.length === 0) {
  emptyDiv.style.display = 'block';
} else {
  history.slice(0, 10).forEach(item => {
    const tr = document.createElement('tr');
    const badgeClass = STATUS_COLOR[item.status] || 'accent';
    tr.innerHTML = `
      <td><span style="font-size:1.2rem">${item.emoji}</span> ${item.name}</td>
      <td>
        <div style="display:flex;align-items:center;gap:0.6rem;">
          <div class="progress-bar" style="width:80px;"><div class="progress-fill ${item.freshness >= 71 ? 'green' : item.freshness >= 41 ? 'orange' : 'red'}" style="width:${item.freshness}%"></div></div>
          <span style="font-size:0.82rem;font-weight:600;">${item.freshness}%</span>
        </div>
      </td>
      <td>${item.shelf > 0 ? item.shelf + ' days' : '<span style="color:var(--danger)">Consume now</span>'}</td>
      <td><span class="badge badge-${badgeClass}">${item.status}</span></td>
      <td style="color:var(--text-muted);font-size:0.8rem;">${formatDate(item.timestamp)}</td>
    `;
    tbody.appendChild(tr);
  });
}
