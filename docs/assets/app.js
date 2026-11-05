(function () {
  var dialog = document.querySelector('dialog.search');
  var input = dialog && dialog.querySelector('input');
  var list = dialog && dialog.querySelector('ul');
  var index = null;
  var selected = 0;

  function load() {
    if (index) return Promise.resolve(index);
    return fetch('search.json').then(function (r) { return r.json(); }).then(function (data) {
      index = data;
      return data;
    });
  }

  function count(hay, needle) {
    var n = 0, at = hay.indexOf(needle);
    while (at !== -1) { n++; at = hay.indexOf(needle, at + needle.length); }
    return n;
  }

  function score(item, q) {
    var title = item.title.toLowerCase();
    var headings = (item.headings || '').toLowerCase();
    var body = item.body.toLowerCase();
    var titleHits = count(title, q);
    var headingHits = count(headings, q);
    var bodyHits = count(body, q);
    if (!titleHits && !headingHits && !bodyHits) return 0;
    // A page named after the query beats a page that merely mentions it.
    return titleHits * 50 + headingHits * 10 + bodyHits;
  }

  function render(items) {
    list.innerHTML = '';
    if (!items.length) {
      list.innerHTML = '<li class="search-empty">Nothing found. Try "player", "seek", "flac".</li>';
      return;
    }
    items.forEach(function (item, i) {
      var li = document.createElement('li');
      var a = document.createElement('a');
      a.href = item.url;
      a.innerHTML = '<b>' + item.title + '</b><span>' + item.section + '</span>';
      if (i === 0) { a.classList.add('sel'); }
      li.appendChild(a);
      list.appendChild(li);
    });
    selected = 0;
  }

  function search(q) {
    load().then(function (data) {
      q = q.trim().toLowerCase();
      if (!q) { render(data.slice(0, 8)); return; }
      var hits = data
        .map(function (item) { return { item: item, s: score(item, q) }; })
        .filter(function (h) { return h.s > 0; })
        .sort(function (a, b) { return b.s - a.s || a.item.title.length - b.item.title.length; })
        .slice(0, 10)
        .map(function (h) { return h.item; });
      render(hits);
    });
  }

  function open() {
    if (!dialog) return;
    if (!dialog.open) dialog.showModal();
    input.value = '';
    search('');
    input.focus();
  }

  document.addEventListener('click', function (event) {
    var trigger = event.target.closest('[data-search]');
    if (trigger) { event.preventDefault(); open(); }
  });

  document.addEventListener('keydown', function (event) {
    var typing = /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName);
    if (event.key === '/' && !typing) { event.preventDefault(); open(); return; }
    if (event.key === 'Escape' && dialog && dialog.open) { dialog.close(); return; }
    if (!dialog || !dialog.open) return;
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      var links = list.querySelectorAll('a');
      if (!links.length) return;
      event.preventDefault();
      selected = (selected + (event.key === 'ArrowDown' ? 1 : links.length - 1)) % links.length;
      links.forEach(function (l, i) { l.classList.toggle('sel', i === selected); });
      links[selected].scrollIntoView({ block: 'nearest' });
    }
    if (event.key === 'Enter') {
      var link = list.querySelector('a.sel');
      if (link) { event.preventDefault(); window.location.href = link.href; }
    }
  });

  if (input) {
    input.addEventListener('input', function () { search(input.value); });
  }

  // copy buttons
  document.querySelectorAll('pre').forEach(function (pre) {
    var button = document.createElement('button');
    button.className = 'copy';
    button.type = 'button';
    button.textContent = 'Copy';
    button.addEventListener('click', function () {
      var code = pre.querySelector('code');
      navigator.clipboard.writeText(code ? code.innerText : pre.innerText).then(function () {
        button.textContent = 'Copied';
        setTimeout(function () { button.textContent = 'Copy'; }, 1400);
      });
    });
    pre.appendChild(button);
  });

  // table of contents
  var tocLinks = Array.prototype.slice.call(document.querySelectorAll('.toc a'));
  var headings = tocLinks
    .map(function (a) { return document.getElementById(a.getAttribute('href').slice(1)); })
    .filter(Boolean);
  if (headings.length && 'IntersectionObserver' in window) {
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        tocLinks.forEach(function (a) {
          a.classList.toggle('active', a.getAttribute('href') === '#' + entry.target.id);
        });
      });
    }, { rootMargin: '-10% 0px -75% 0px' });
    headings.forEach(function (h) { observer.observe(h); });
  }
})();
