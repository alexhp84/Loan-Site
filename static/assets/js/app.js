document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("loan-form");
    const resYes = document.getElementById("result-yes");
    const resNo  = document.getElementById("result-no");
    const themeBtn = document.getElementById("theme-toggle");

    if (themeBtn) {
        const setTheme = t => {
            document.documentElement.dataset.theme = t;
            localStorage.setItem("theme", t);
        };
        const saved = localStorage.getItem("theme") || "light";
        setTheme(saved);
        themeBtn.addEventListener("click", () => {
            setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
        });
    }

    if (form) {
        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            const data = Object.fromEntries(new FormData(form).entries());

            ["person_income", "loan_amnt", "loan_int_rate", "loan_percent_income", "credit_score"]
                .forEach(k => data[k] = Number(data[k]));

            try {
                const resp = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(data)
                });
                const json = await resp.json();
                if (json.prediction_code === 0 || json.prediction === "Approved") {
                    if (resYes) resYes.classList.remove("hidden");
                    confetti();
                } else {
                    if (resNo) resNo.classList.remove("hidden");
                }
                form.classList.add("hidden");
            } catch(err) {
                alert("Error: " + err);
            }
        });
    }

    const retryBtn = document.getElementById("retry-application");
    if (retryBtn) {
        retryBtn.onclick = () => {
            if (form) {
                form.reset();
                form.classList.remove("hidden");
            }
            if (resYes) resYes.classList.add("hidden");
            if (resNo) resNo.classList.add("hidden");
        };
    }

    function confetti() {
        const c = document.createElement("canvas");
        c.style.position = "fixed"; c.style.top = 0; c.style.left = 0;
        c.style.width = "100%"; c.style.height = "100%";
        c.style.pointerEvents = "none";
        document.body.appendChild(c);
        const ctx = c.getContext("2d");
        const W = c.width = window.innerWidth;
        const H = c.height = window.innerHeight;
        const particles = [];
        for (let i = 0; i < 120; i++) {
            particles.push({
                x: W / 2, y: H / 2, r: Math.random() * 6 + 2,
                color: `hsl(${Math.random() * 360},60%,55%)`,
                vx: Math.random() * 6 - 3, vy: Math.random() * -6 - 2, life: Math.random() * 30 + 30
            });
        }
        function draw() {
            ctx.clearRect(0, 0, W, H);
            particles.forEach(p => {
                ctx.fillStyle = p.color;
                ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 2 * Math.PI); ctx.fill();
                p.x += p.vx; p.y += p.vy; p.vy += 0.15; p.life--;
            });
            if (particles.some(p => p.life > 0)) requestAnimationFrame(draw);
            else c.remove();
        }
        draw();
    }

    if (window.lucide) lucide.createIcons();
});
