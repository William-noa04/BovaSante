// BovaSanté — cache du shell PWA; les appels backend restent séparés et peuvent être synchronisés ensuite.
//
// Les pages HTML (navigation) passent en network-first : sinon, une fois "/" mis en
// cache, un nouveau déploiement (nouveaux noms de fichiers JS/CSS hashés) reste invisible
// et le SW continue de servir l'ancien index.html, qui pointe vers des assets supprimés
// (page blanche tant qu'on ne change pas d'URL). Les assets statiques restent cache-first :
// leurs noms sont hashés par Vite, donc une même URL = un contenu immuable.
const CACHE_NAME = "bovasante-shell-v2";
const APP_SHELL = ["/manifest.json"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;

  if (event.request.mode === "navigate") {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          const copy = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copy));
          return response;
        })
        .catch(() => caches.match(event.request).then((cached) => cached || caches.match("/")))
    );
    return;
  }

  event.respondWith(
    caches.match(event.request).then(
      (cached) =>
        cached ||
        fetch(event.request).then((response) => {
          const copy = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copy));
          return response;
        })
    )
  );
});
