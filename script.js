/* Neuron Enterprise Documentation — interactions */

(function () {
    'use strict';

    // Lucide icons (optional)
    if (typeof lucide !== 'undefined') lucide.createIcons();

    // Neural background
    const canvas = document.getElementById('neural-bg');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        let particles = [];
        const particleCount = 80;
        const connectionDistance = 140;

        function initCanvas() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            particles = [];
            for (let i = 0; i < particleCount; i++) {
                particles.push({
                    x: Math.random() * canvas.width,
                    y: Math.random() * canvas.height,
                    vx: (Math.random() - 0.5) * 0.4,
                    vy: (Math.random() - 0.5) * 0.4,
                    size: Math.random() * 1.5 + 0.5
                });
            }
        }

        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            particles.forEach((p, i) => {
                p.x += p.vx;
                p.y += p.vy;
                if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
                if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
                ctx.fillStyle = 'rgba(63, 185, 80, 0.35)';
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fill();
                for (let j = i + 1; j < particles.length; j++) {
                    const p2 = particles[j];
                    const dist = Math.hypot(p.x - p2.x, p.y - p2.y);
                    if (dist < connectionDistance) {
                        ctx.strokeStyle = `rgba(63, 185, 80, ${0.08 * (1 - dist / connectionDistance)})`;
                        ctx.beginPath();
                        ctx.moveTo(p.x, p.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.stroke();
                    }
                }
            });
            requestAnimationFrame(animate);
        }

        window.addEventListener('resize', initCanvas);
        initCanvas();
        animate();
    }

    // Copy buttons on code blocks
    document.querySelectorAll('.code-example').forEach((block) => {
        const header = block.querySelector('.code-header');
        const pre = block.querySelector('pre code');
        if (!header || !pre) return;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'copy-btn';
        btn.textContent = 'Copy';
        btn.addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(pre.textContent);
                btn.textContent = 'Copied';
                btn.classList.add('copied');
                setTimeout(() => {
                    btn.textContent = 'Copy';
                    btn.classList.remove('copied');
                }, 2000);
            } catch (_) {
                btn.textContent = 'Failed';
            }
        });
        header.appendChild(btn);
    });

    // Scroll spy
    const sections = document.querySelectorAll('section[id], .doc-block[id]');
    const navLinks = document.querySelectorAll('.nav-list a');

    function updateActiveNav() {
        let current = '';
        const offset = 120;
        sections.forEach((section) => {
            const top = section.getBoundingClientRect().top;
            if (top <= offset) current = section.id;
        });
        navLinks.forEach((link) => {
            link.classList.remove('active');
            const href = link.getAttribute('href');
            if (href === '#' + current) link.classList.add('active');
        });
    }

    window.addEventListener('scroll', updateActiveNav, { passive: true });
    updateActiveNav();

    // Smooth scroll
    navLinks.forEach((link) => {
        link.addEventListener('click', (e) => {
            const href = link.getAttribute('href');
            if (!href || !href.startsWith('#')) return;
            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                closeMobileNav();
            }
        });
    });

    // Mobile nav
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.querySelector('.sidebar-overlay');
    const toggle = document.querySelector('.mobile-nav-toggle');

    function openMobileNav() {
        sidebar?.classList.add('open');
        overlay?.classList.add('visible');
    }
    function closeMobileNav() {
        sidebar?.classList.remove('open');
        overlay?.classList.remove('visible');
    }

    toggle?.addEventListener('click', () => {
        sidebar?.classList.contains('open') ? closeMobileNav() : openMobileNav();
    });
    overlay?.addEventListener('click', closeMobileNav);

    // Full-text search (Fuse.js loaded from CDN)
    const searchInput = document.getElementById('doc-search');
    const searchResults = document.getElementById('search-results');

    function buildSearchIndex() {
        const items = [];
        document.querySelectorAll('section[id], .doc-block[id]').forEach((section) => {
            const id = section.id;
            const titleEl = section.querySelector('.page-title, .section-title');
            const title = titleEl ? titleEl.textContent.replace(/¶.*/, '').trim() : id;
            const body = section.innerText.slice(0, 2000);
            items.push({ id, title, body, href: '#' + id });
            section.querySelectorAll('.subsection-title, .method-sig').forEach((el) => {
                const sub = el.textContent.trim().slice(0, 120);
                if (sub.length > 3) {
                    items.push({
                        id: id + '-' + sub.slice(0, 30).replace(/\W+/g, '-'),
                        title: sub,
                        body: el.closest('.method-block')?.innerText?.slice(0, 800) || sub,
                        href: '#' + id
                    });
                }
            });
        });
        document.querySelectorAll('.exhaustive-table code').forEach((code) => {
            const env = code.textContent.trim();
            if (env === env.toUpperCase() && env.includes('_')) {
                const row = code.closest('tr');
                items.push({
                    id: 'env-' + env,
                    title: env,
                    body: row ? row.innerText : env,
                    href: '#configuration'
                });
            }
        });
        return items;
    }

    let fuse = null;
    function initSearch() {
        if (typeof Fuse === 'undefined') return;
        const index = buildSearchIndex();
        fuse = new Fuse(index, {
            keys: ['title', 'body'],
            threshold: 0.35,
            includeScore: true,
            minMatchCharLength: 2
        });
    }

    function renderSearchResults(results) {
        if (!searchResults) return;
        searchResults.innerHTML = '';
        if (!results.length) {
            searchResults.classList.remove('visible');
            return;
        }
        results.slice(0, 12).forEach(({ item }) => {
            const a = document.createElement('a');
            a.className = 'search-result-item';
            a.href = item.href;
            a.innerHTML = '<strong>' + escapeHtml(item.title) + '</strong>' + escapeHtml(item.body.slice(0, 80)) + '…';
            a.addEventListener('click', (e) => {
                e.preventDefault();
                document.querySelector(item.href)?.scrollIntoView({ behavior: 'smooth' });
                searchResults.classList.remove('visible');
                searchInput.value = '';
                closeMobileNav();
            });
            searchResults.appendChild(a);
        });
        searchResults.classList.add('visible');
    }

    function escapeHtml(s) {
        const d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }

    searchInput?.addEventListener('input', () => {
        if (!fuse) initSearch();
        const q = searchInput.value.trim();
        if (q.length < 2) {
            searchResults?.classList.remove('visible');
            return;
        }
        renderSearchResults(fuse.search(q));
    });

    searchInput?.addEventListener('focus', () => {
        if (!fuse) initSearch();
    });

    document.addEventListener('click', (e) => {
        if (!searchInput?.contains(e.target) && !searchResults?.contains(e.target)) {
            searchResults?.classList.remove('visible');
        }
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === '/' && document.activeElement !== searchInput) {
            e.preventDefault();
            searchInput?.focus();
        }
        if (e.key === 'Escape') {
            searchResults?.classList.remove('visible');
            searchInput?.blur();
            closeMobileNav();
        }
    });

    // Mermaid
    if (typeof mermaid !== 'undefined') {
        mermaid.initialize({
            startOnLoad: true,
            theme: 'dark',
            securityLevel: 'loose',
            flowchart: { curve: 'basis' }
        });
    }

    initSearch();
})();
