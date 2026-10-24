(function () {
  const LEAGUES = [["E0", "Premier League"], ["SP1", "La Liga"], ["D1", "Bundesliga"],
                   ["I1", "Serie A"], ["F1", "Ligue 1"]];
  const cur = new URLSearchParams(location.search).get("league") || "E0";
  const nav = document.createElement("nav");
  nav.className = "topnav";

  const brand = document.createElement("a");
  brand.className = "brand";
  brand.href = "index.html?league=" + cur;
  brand.textContent = "Football Predictor";
  nav.appendChild(brand);

  const onTrackRecord = location.pathname.endsWith("track-record.html");

  LEAGUES.forEach(([code, name]) => {
    const a = document.createElement("a");
    a.href = "index.html?league=" + code;
    a.textContent = name;
    if (code === cur) a.className = "active";
    nav.appendChild(a);
  });

  const tr = document.createElement("a");
  tr.href = "track-record.html?league=" + cur;
  tr.textContent = "Track record";
  tr.className = "tr-link" + (onTrackRecord ? " active" : "");
  nav.appendChild(tr);

  document.body.insertBefore(nav, document.body.firstChild);
})();