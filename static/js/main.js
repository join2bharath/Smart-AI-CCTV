/**
 * main.js — College Management System helper scripts
 */

// ── Sidebar toggle ─────────────────────────────────────────────────────────
function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('sidebarOverlay');
  if (!sidebar) return;
  sidebar.classList.toggle('open');
  overlay.classList.toggle('open');
  document.body.style.overflow = sidebar.classList.contains('open') ? 'hidden' : '';
}

// ── Auto-dismiss flash messages ────────────────────────────────────────────
(function autoFlash() {
  setTimeout(() => {
    document.querySelectorAll('.flash').forEach(el => {
      el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
      el.style.opacity = '0';
      el.style.transform = 'translateX(40px)';
      setTimeout(() => el.remove(), 500);
    });
  }, 5000);
})();

// ── Mark input live P/F preview ────────────────────────────────────────────
document.addEventListener('input', function(e) {
  if (e.target.classList.contains('marks-input')) {
    const val = parseInt(e.target.value, 10);
    const cell = e.target.closest('td');
    if (!cell) return;
    let badge = cell.querySelector('.live-pf');
    if (!badge) {
      badge = document.createElement('span');
      badge.className = 'live-pf pf-badge';
      cell.appendChild(badge);
    }
    if (!isNaN(val)) {
      badge.textContent = val >= 35 ? 'P' : 'F';
      badge.className = `live-pf pf-badge ${val >= 35 ? 'pass' : 'fail'}`;
    }
  }
});

// ── Attendance toggle buttons ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.att-toggle').forEach(function(group) {
    group.querySelectorAll('.att-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        const parent = btn.closest('.att-toggle');
        parent.querySelectorAll('.att-btn').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
        const hiddenInput = parent.closest('.att-student-row').querySelector('input[type=hidden]');
        if (hiddenInput) hiddenInput.value = btn.dataset.val;
      });
    });
  });

  // Confirm submit for attendance form
  const attForm = document.getElementById('attendanceForm');
  if (attForm) {
    attForm.addEventListener('submit', function(e) {
      const btn = attForm.querySelector('[type=submit]');
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa fa-spinner fa-spin"></i> Saving…';
      }
    });
  }

  // Marks form submit spinner
  const marksForm = document.getElementById('marksForm');
  if (marksForm) {
    marksForm.addEventListener('submit', function() {
      const btn = marksForm.querySelector('[type=submit]');
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa fa-spinner fa-spin"></i> Saving…';
      }
    });
  }
});

// ── Live search filter for tables ──────────────────────────────────────────
function liveSearch(inputId, tableId) {
  const input = document.getElementById(inputId);
  const table = document.getElementById(tableId);
  if (!input || !table) return;
  input.addEventListener('input', function() {
    const q = this.value.toLowerCase();
    table.querySelectorAll('tbody tr').forEach(tr => {
      tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none';
    });
  });
}

// ── Alert acknowledge via AJAX ─────────────────────────────────────────────
function ackAlert(alertId, apiBase) {
  fetch(`${apiBase}${alertId}`, { method: 'POST', headers: { 'X-Requested-With': 'XMLHttpRequest' } })
    .then(r => r.json())
    .then(() => {
      const row = document.getElementById(`alert-row-${alertId}`);
      if (row) {
        row.classList.remove('unread');
        const ackBtn = row.querySelector('.ack-btn');
        if (ackBtn) { ackBtn.disabled = true; ackBtn.textContent = '✓ Done'; }
      }
    })
    .catch(() => {});
}

// ── Dept card click navigation ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.dept-card[data-href]').forEach(card => {
    card.addEventListener('click', function() {
      window.location.href = this.dataset.href;
    });
  });
});
