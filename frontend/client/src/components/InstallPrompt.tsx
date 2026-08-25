import { useEffect, useState } from "react";
import { Download, Share, X } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";

const DISMISS_KEY = "bovasante:install-dismissed";

interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed" }>;
}

declare global {
  interface Window {
    __bovaInstallPrompt?: BeforeInstallPromptEvent;
  }
}

function isStandalone() {
  return window.matchMedia("(display-mode: standalone)").matches
    || (window.navigator as Navigator & { standalone?: boolean }).standalone === true;
}
function isIos() {
  return /iphone|ipad|ipod/i.test(navigator.userAgent)
    || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
}

export function InstallPrompt() {
  const { t } = useLanguage();
  const [deferred, setDeferred] = useState<BeforeInstallPromptEvent | null>(null);
  const [showIosHint, setShowIosHint] = useState(false);
  const [dismissed, setDismissed] = useState(() => localStorage.getItem(DISMISS_KEY) === "1");

  useEffect(() => {
    if (isStandalone()) return;
    if (isIos()) {
      setShowIosHint(true);
      return;
    }
    if (window.__bovaInstallPrompt) setDeferred(window.__bovaInstallPrompt);
    function onBeforeInstall(event: Event) {
      event.preventDefault();
      const promptEvent = event as BeforeInstallPromptEvent;
      window.__bovaInstallPrompt = promptEvent;
      setDeferred(promptEvent);
    }
    function onInstalled() {
      setDeferred(null);
      setDismissed(true);
    }
    window.addEventListener("beforeinstallprompt", onBeforeInstall);
    window.addEventListener("appinstalled", onInstalled);
    return () => {
      window.removeEventListener("beforeinstallprompt", onBeforeInstall);
      window.removeEventListener("appinstalled", onInstalled);
    };
  }, []);

  if (dismissed || (!deferred && !showIosHint)) return null;

  async function install() {
    if (!deferred) return;
    await deferred.prompt();
    const choice = await deferred.userChoice;
    setDeferred(null);
    if (choice.outcome === "accepted") setDismissed(true);
  }

  function dismiss() {
    localStorage.setItem(DISMISS_KEY, "1");
    setDismissed(true);
  }

  return (
    <div className="install-banner-wrap">
      <div className={`install-banner ${showIosHint ? "ios" : ""}`}>
        <span className="install-banner-icon">{showIosHint ? <Share /> : <Download />}</span>
        <div>
          <strong>{t("install.title")}</strong>
          <small>{showIosHint ? t("install.body.ios") : t("install.body")}</small>
        </div>
        <div className="install-banner-actions">
          {!showIosHint && <button className="button primary" onClick={install}>{t("install.cta")}</button>}
          <button className="install-banner-close" onClick={dismiss} aria-label={t("install.close")}>
            <X />
          </button>
        </div>
      </div>
    </div>
  );
}
