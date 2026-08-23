const translations = {
    en: {
        site_title: "Loan Approval Checker",
        nav_home: "Home",
        nav_dashboard: "Dashboard",
        nav_model: "Model",
        nav_api: "API",
        nav_logs: "Logs"
    },
    he: {
        site_title: "בדיקת אישור הלוואה",
        nav_home: "בית",
        nav_dashboard: "לוח בקרה",
        nav_model: "מודל",
        nav_api: "ממשק API",
        nav_logs: "יומנים"
    },
    ar: {
        site_title: "אداة التحقق من الموافقة على القرض",
        nav_home: "الرئيسية",
        nav_dashboard: "لوحة التحكم",
        nav_model: "النموذج",
        nav_api: "واجهة البرمجة",
        nav_logs: "السجلات"
    }
};

function applyLanguage(lang) {
    const selectedLang = translations[lang] ? lang : 'en';
    const langData = translations[selectedLang];

    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (langData[key]) {
            el.textContent = langData[key];
        }
    });

    document.documentElement.lang = selectedLang;
    document.documentElement.dir = (selectedLang === 'he' || selectedLang === 'ar') ? 'rtl' : 'ltr';
    localStorage.setItem('preferred_lang', selectedLang);
}

document.addEventListener('DOMContentLoaded', () => {
    const langSelect = document.getElementById('lang-select');
    const savedLang = localStorage.getItem('preferred_lang') || 'en';

    if (langSelect) {
        langSelect.value = savedLang;
        langSelect.addEventListener('change', (e) => applyLanguage(e.target.value));
    }

    applyLanguage(savedLang);
});