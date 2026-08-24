import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

createRoot(document.getElementById("root")!).render(<App />);

// Le service worker ne doit s'enregistrer que sur le build de production :
// en dev, son cache-first entre en conflit avec le hot-reload de Vite et
// finit par servir une version figée de l'app (indépendamment par onglet/profil).
if (import.meta.env.PROD && "serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("/sw.js"));
}
