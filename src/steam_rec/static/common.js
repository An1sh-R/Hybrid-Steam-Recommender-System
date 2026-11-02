/*
  Helpers shared by both pages (home.html and recommend.html), loaded with a plain <script> tag (D41).
  - element / gameImage / formatPrice / formatCount: building cards
  - loadPicks / savePicks: the games picked so far, kept while you move between the two pages
*/

const MIN_GAMES = 3;
const MAX_GAMES = 5;
const STEAM_IMAGES = "https://cdn.akamai.steamstatic.com/steam/apps";
const PICKS_KEY = "steam-rec-picks";

// Build an element with optional class and text. textContent (not innerHTML) so game titles
// can never be interpreted as HTML.
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// A game's picture from Steam's image server. `file` is "header.jpg" (460 x 215) or
// "capsule_sm_120.jpg" (120 x 45). If Steam has no image, swap in a plain placeholder box.
function gameImage(appId, file, className) {
  const image = element("img", className);
  image.src = `${STEAM_IMAGES}/${appId}/${file}`;
  image.alt = "";  // decorative: the title is always shown next to it
  image.loading = "lazy";
  image.addEventListener("error", () => {
    const placeholder = element("div", `${className} no-image`, className === "cover" ? "No image" : "");
    image.replaceWith(placeholder);
  }, { once: true });
  return image;
}

// A title linking to the game's Steam store page, opened in a new tab.
function storeLink(appId, title) {
  const link = element("a", null, title);
  link.href = `https://store.steampowered.com/app/${appId}`;
  link.target = "_blank";
  link.rel = "noopener";
  return link;
}

function formatPrice(price) {
  if (price === 0) return "Free";
  return `$${price.toFixed(2)}`;
}

// 1234567 -> "1.2M", 45600 -> "46K", 950 -> "950".
function formatCount(count) {
  if (count >= 1000000) return `${(count / 1000000).toFixed(1)}M`;
  if (count >= 1000) return `${Math.round(count / 1000)}K`;
  return String(count);
}

// The picked games, [{app_id, title, release_year}, ...], kept in this browser tab's sessionStorage
// so they survive going from the home page to the recommendations page and back. Storage can be
// unavailable (e.g. blocked by privacy settings); then picks simply aren't remembered.
function loadPicks() {
  try {
    const saved = sessionStorage.getItem(PICKS_KEY);
    if (saved === null) return [];
    return JSON.parse(saved);
  } catch (error) {
    return [];
  }
}

function savePicks(picks) {
  const plain = [];
  for (const pick of picks) {
    plain.push({ app_id: pick.app_id, title: pick.title, release_year: pick.release_year });
  }
  try {
    sessionStorage.setItem(PICKS_KEY, JSON.stringify(plain));
  } catch (error) {
    // Not remembered; the page still works.
  }
}
