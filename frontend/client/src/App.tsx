import { Link, Route, Switch, useLocation } from "wouter";
import { Activity, Bot, Home, Languages, Moon, PlusCircle, Stethoscope, Sun } from "lucide-react";
import { Assistant, Analysis, CattleDetails, Dashboard, History, NotFound } from "@/pages/Pages";
import { Veterinarians } from "@/pages/Veterinarians";
import { InstallPrompt } from "@/components/InstallPrompt";
import { ThemeProvider, useTheme } from "@/contexts/ThemeContext";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";

const nav = [
  ["/", "nav.home", "nav.home.short", Home],
  ["/analyse", "nav.analyse", "nav.analyse.short", PlusCircle],
  ["/assistant", "nav.assistant", "nav.assistant", Bot],
  ["/veterinaires", "nav.vets", "nav.vets", Stethoscope],
  ["/historique", "nav.history", "nav.history", Activity],
] as const;

function SidebarControls() {
  const { theme, toggleTheme } = useTheme();
  const { lang, toggleLang, t } = useLanguage();
  return (
    <div className="sidebar-controls">
      <button
        type="button"
        className="sidebar-toggle"
        onClick={toggleTheme}
        aria-label={theme === "dark" ? t("sidebar.theme.light") : t("sidebar.theme.dark")}
      >
        {theme === "dark" ? <Sun /> : <Moon />}
      </button>
      <button type="button" className="sidebar-toggle lang-toggle" onClick={toggleLang} aria-label="FR / EN">
        <Languages /><span>{lang.toUpperCase()}</span>
      </button>
    </div>
  );
}

function isActive(location: string, href: string) {
  if (href === "/analyse") return location === "/analyse" || location === "/diagnostic-rapide";
  return location === href;
}

function Shell() {
  const [location] = useLocation();
  const { t } = useLanguage();
  return <div className="app-shell"><SidebarControls/><aside className="sidebar"><Link className="brand" href="/"><span className="brand-mark"><img src="/logo.png" alt="BovaSanté"/></span><span><b>BovaSanté</b><small>{t("brand.tagline")}</small></span></Link><nav>{nav.map(([href, labelKey, , Icon]) => <Link key={href} href={href} className={isActive(location, href) ? "active" : ""}><Icon/>{t(labelKey)}</Link>)}</nav><div className="service-status"><i/><span>{t("sidebar.status")}<br/><small>{t("sidebar.statusHint")}</small></span></div></aside><main><InstallPrompt/><Switch><Route path="/" component={Dashboard}/><Route path="/analyse" component={Analysis}/><Route path="/diagnostic-rapide" component={Analysis} /><Route path="/assistant" component={Assistant}/><Route path="/veterinaires" component={Veterinarians}/><Route path="/historique" component={History}/><Route path="/bovins/:id" component={CattleDetails}/><Route component={NotFound}/></Switch></main><nav className="mobile-nav">{nav.slice(0, 4).map(([href, , shortKey, Icon]) => <Link key={href} href={href} className={isActive(location, href) ? "active" : ""}><Icon/><span>{t(shortKey)}</span></Link>)}</nav></div>;
}

export default function App() {
  return (
    <ThemeProvider switchable defaultTheme="light">
      <LanguageProvider>
        <Shell />
      </LanguageProvider>
    </ThemeProvider>
  );
}
