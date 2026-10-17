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

  LEAGUES.forEach(([code, name]) => {
    const a = document.createElement("a");
    a.href = "index.html?league=" + code;
    a.textContent = name;
    if (code === cur) a.className = "active";
    nav.appendChild(a);
  });
  document.body.insertBefore(nav, document.body.firstChild);
})();