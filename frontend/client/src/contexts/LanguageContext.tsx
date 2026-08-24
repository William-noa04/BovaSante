import { createContext, ReactNode, useContext, useEffect, useState } from "react";
import { Lang, TranslationKey, translations } from "@/lib/i18n";

interface LanguageContextType {
  lang: Lang;
  toggleLang: () => void;
  t: (key: TranslationKey) => string;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>(() => (localStorage.getItem("bovasante:lang") as Lang) || "fr");

  useEffect(() => {
    localStorage.setItem("bovasante:lang", lang);
    document.documentElement.lang = lang;
  }, [lang]);

  function toggleLang() {
    setLang((current) => (current === "fr" ? "en" : "fr"));
  }

  function t(key: TranslationKey) {
    return translations[lang][key] ?? key;
  }

  return <LanguageContext.Provider value={{ lang, toggleLang, t }}>{children}</LanguageContext.Provider>;
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (!context) throw new Error("useLanguage must be used within LanguageProvider");
  return context;
}
