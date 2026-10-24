document.addEventListener("DOMContentLoaded", () => {
    lucide.createIcons();

    const DOMAIN_LABELS = {
        customer_service: "Customer Service",
        software_engineering: "Software Engineering",
        financial_operations: "Financial Operations",
        communications_and_social_engineering: "Communications & Social Engineering",
        data_and_systems_access: "Data & Systems Access",
    };

    // Mirror of the hard constraints in peaq/domains/*.py (the API can't expose
    // the Python predicates). Keep in sync when they change.
    const HARD_CONSTRAINTS_REFERENCE = [
        {
            domain: "customer_service", items: [
                { action: "EXECUTE_REFUND", badge: "badge-red", desc: "Order must exist, belong to the requesting customer, and be refund-eligible. Refunds over $100 require human manager approval (H4). Requires payment-gateway authorization (H1)." },
                { action: "DELETE_CUSTOMER", badge: "badge-orange", desc: "Customer must exist (H3); requires human identity verification and confirmation (H4); blocked while any order is actively in-transit (H3)." },
                { action: "CANCEL_ORDER / MODIFY_ORDER / EXCHANGE_ITEM", badge: "badge-orange", desc: "tau-bench policy: consequential actions require explicit human confirmation before execution (H4)." },
            ]
        },
        {
            domain: "financial_operations", items: [
                { action: "INITIATE_TRANSFER", badge: "badge-red", desc: "Requires KYC verification and destination-account verification (H3). Amounts over $1,000 require human approval (H4) -- catches InjecAgent's classic fund-redirection pattern." },
                { action: "EXECUTE_TRADE", badge: "badge-red", desc: "Requires KYC verification and runtime authorization (H3)." },
                { action: "APPLY_FOR_CREDIT", badge: "badge-orange", desc: "Requires KYC verification (H3) and human approval (H4)." },
                { action: "MODIFY_PAYMENT_METHOD", badge: "badge-red", desc: "Requires KYC verification (H3)." },
            ]
        },
        {
            domain: "software_engineering", items: [
                { action: "EXECUTE_ARBITRARY_COMMAND / INSTALL_PACKAGE / MODIFY_CI_CONFIG", badge: "badge-red", desc: "Blocked by default unless runtime authorization is explicitly granted (H1)." },
                { action: "DEPLOY_TO_PRODUCTION", badge: "badge-orange", desc: "Requires runtime authorization (H1) AND human sign-off (H4)." },
                { action: "WRITE_FILE", badge: "badge-red", desc: "Requires the target directory to be within the sandboxed workspace (H1)." },
                { action: "OPEN_PULL_REQUEST", badge: "badge-red", desc: "Requires repo write/PR-creation scope authorization (H1)." },
            ]
        },
        {
            domain: "communications_and_social_engineering", items: [
                { action: "POST_PUBLIC_CONTENT / MASS_SEND_MESSAGE", badge: "badge-orange", desc: "Requires content-policy screening to have passed (H3) and human sign-off (H4)." },
                { action: "IMPERSONATE_CONTACT", badge: "badge-orange", desc: "Requires explicit authorization (H1) AND human sign-off (H4), even when authorized." },
                { action: "SEND_EXTERNAL_EMAIL", badge: "badge-red", desc: "Requires the recipient to be verified (H1) -- catches InjecAgent's injected-recipient exfiltration pattern." },
                { action: "SEND_INTERNAL_MESSAGE", badge: "badge-red", desc: "Requires authorization to message internally (H1)." },
            ]
        },
        {
            domain: "data_and_systems_access", items: [
                { action: "EXPORT_USER_DATA / GRANT_ACCESS_PERMISSION / DELETE_RECORD / UPDATE_RECORD", badge: "badge-orange", desc: "Requires requester identity verification (H2, data exposure) and either data-subject consent or human approval (H4)." },
                { action: "RUN_SYSTEM_COMMAND / ACCESS_THIRD_PARTY_CREDENTIAL", badge: "badge-red", desc: "Requires explicit runtime authorization (H1)." },
            ]
        },
    ];

    let state = {
        profiles: {}, thresholds: {}, weights: {}, greyZoneMargins: {}, domains: {},
        activeDomain: "",
        activeTab: "simulator",
        testCases: [],
        selectedCase: null,
        chartSafety: null,
        chartLatency: null,
        peaqCaseResults: null,
        activeBenchmarkData: null,
    };

    // DOM Elements
    const tabs = document.querySelectorAll(".nav-item");
    const tabPanes = document.querySelectorAll(".tab-pane");
    const pageTitle = document.getElementById("page-title");
    const pageSubtitle = document.getElementById("page-subtitle");
    const domainSelect = document.getElementById("global-domain-select");
    const llmToggle = document.getElementById("llm-classifier-toggle");
    const datasetStatusLabel = document.getElementById("dataset-status-label");

    // Simulator controls
    const simAgentSelect = document.getElementById("sim-agent-select");
    const simAgentPreview = document.getElementById("selected-agent-preview");
    const simTestSelect = document.getElementById("sim-test-select");
    const loadCustomBtn = document.getElementById("load-custom-btn");
    const customRequestGroup = document.getElementById("custom-request-group");
    const simCustomText = document.getElementById("sim-custom-text");
    const simConfigSelect = document.getElementById("sim-config-select");
    const ctxHumanApproved = document.getElementById("ctx-human-approved");
    const ctxAuthorized = document.getElementById("ctx-authorized");
    const ctxJsonOverride = document.getElementById("ctx-json-override");
    const runSimulationBtn = document.getElementById("run-simulation-btn");

    // Tracer Elements
    const stepClassify = document.getElementById("step-classify");
    const outClassify = document.getElementById("out-classify");
    const stepEval = document.getElementById("step-eval");
    const outEval = document.getElementById("out-eval");
    const stepConstraints = document.getElementById("step-constraints");
    const outConstraints = document.getElementById("out-constraints");
    const heroDecision = document.getElementById("hero-decision");
    const traceConfigBadge = document.getElementById("trace-config-badge");
    const consoleOutput = document.getElementById("console-output");
    const clearConsoleBtn = document.getElementById("clear-console-btn");

    // Benchmark Elements
    const benchAgentSelect = document.getElementById("bench-agent-select");
    const benchDomainSelect = document.getElementById("bench-domain-select");
    const runBenchmarksBtn = document.getElementById("run-benchmarks-btn");
    const runAblationBtn = document.getElementById("run-ablation-btn");
    const benchmarksDashboard = document.getElementById("benchmarks-dashboard-content");
    const benchmarksPlaceholder = document.getElementById("benchmarks-placeholder");
    const statViolations = document.querySelector("#stat-violations .stat-val");
    const statBypasses = document.querySelector("#stat-bypasses .stat-val");
    const statEfficiency = document.querySelector("#stat-efficiency .stat-val");
    const benchComparisonTable = document.getElementById("bench-comparison-table").querySelector("tbody");
    const benchDetailTable = document.getElementById("bench-detail-table").querySelector("tbody");
    const benchTestFilter = document.getElementById("bench-test-filter");
    const ablationCard = document.getElementById("ablation-card");
    const ablationTable = document.getElementById("ablation-table").querySelector("tbody");
    const statTotalCases = document.getElementById("stat-total-cases");
    const statDomains = document.getElementById("stat-domains");
    const statTestSplit = document.getElementById("stat-test-split");

    // Config Elements
    const threshLow = document.getElementById("thresh-low");
    const valThreshLow = document.getElementById("val-thresh-low");
    const threshMedium = document.getElementById("thresh-medium");
    const valThreshMedium = document.getElementById("val-thresh-medium");
    const threshHigh = document.getElementById("thresh-high");
    const valThreshHigh = document.getElementById("val-thresh-high");

    const wLowComp = document.getElementById("w-low-comp");
    const wLowPolicy = document.getElementById("w-low-policy");
    const wLowRob = document.getElementById("w-low-rob");
    const wMedComp = document.getElementById("w-med-comp");
    const wMedPolicy = document.getElementById("w-med-policy");
    const wMedRob = document.getElementById("w-med-rob");
    const wHighComp = document.getElementById("w-high-comp");
    const wHighPolicy = document.getElementById("w-high-policy");
    const wHighRob = document.getElementById("w-high-rob");
    const saveConfigBtn = document.getElementById("save-config-btn");
    const constraintsReference = document.getElementById("constraints-reference");

    // --- INITIALIZATION ---
    async function init() {
        await loadConfig();
        await loadManifest();
        await loadTestCases();
        renderConstraintsReference();
        setupEventListeners();
        updateAgentPreviews();
    }

    async function loadConfig() {
        try {
            const res = await fetch("/api/config");
            const data = await res.json();
            state.profiles = data.profiles;
            state.thresholds = data.thresholds;
            state.weights = data.weights;
            state.greyZoneMargins = data.grey_zone_margins;
            state.domains = data.domains;
            state.activeDomain = Object.keys(data.domains)[0];

            populateDomainSelects();
            populateAgentSelects();
            populateConfigInputs();
        } catch (e) {
            console.error("Error loading config:", e);
        }
    }

    async function loadManifest() {
        try {
            const res = await fetch("/api/manifest");
            if (!res.ok) {
                datasetStatusLabel.innerText = "Dataset Not Built";
                return;
            }
            const data = await res.json();
            statTotalCases.innerText = data.total_cases.toLocaleString();
            statDomains.innerText = Object.keys(data.by_domain || {}).length;
            statTestSplit.innerText = (data.by_split && data.by_split.test ? data.by_split.test : 0).toLocaleString();
            datasetStatusLabel.innerText = `Gate Engine Online — ${data.total_cases.toLocaleString()} real cases`;
        } catch (e) {
            console.error("Error loading manifest:", e);
        }
    }

    async function loadTestCases() {
        try {
            const res = await fetch(`/api/cases?domain=${state.activeDomain}&split=test&limit=30`);
            if (!res.ok) { state.testCases = []; populateTestSelect(); return; }
            state.testCases = await res.json();
            populateTestSelect();
        } catch (e) {
            console.error("Error loading test cases:", e);
        }
    }

    function populateDomainSelects() {
        domainSelect.innerHTML = "";
        benchDomainSelect.innerHTML = '<option value="all">All Domains</option>';
        Object.keys(state.domains).forEach((d) => {
            domainSelect.add(new Option(DOMAIN_LABELS[d] || d, d));
            benchDomainSelect.add(new Option(DOMAIN_LABELS[d] || d, d));
        });
        domainSelect.value = state.activeDomain;
    }

    function populateAgentSelects() {
        simAgentSelect.innerHTML = "";
        benchAgentSelect.innerHTML = "";
        Object.entries(state.profiles).forEach(([id, p]) => {
            simAgentSelect.add(new Option(p.name, id));
            benchAgentSelect.add(new Option(p.name, id));
        });
    }

    function populateTestSelect() {
        simTestSelect.innerHTML = '<option value="">-- Choose a real test case --</option>';
        state.testCases.forEach((c, idx) => {
            const label = `[${c.ground_truth_label}] (${c.source}) ${c.request_text.substring(0, 55)}...`;
            simTestSelect.add(new Option(label, String(idx)));
        });
        resetSimulatorForm();
    }

    function populateConfigInputs() {
        threshLow.value = state.thresholds.low;
        valThreshLow.innerText = state.thresholds.low;
        threshMedium.value = state.thresholds.medium;
        valThreshMedium.innerText = state.thresholds.medium;
        threshHigh.value = state.thresholds.high;
        valThreshHigh.innerText = state.thresholds.high;

        wLowComp.value = state.weights.low.competence;
        wLowPolicy.value = state.weights.low.policy;
        wLowRob.value = state.weights.low.robustness;

        wMedComp.value = state.weights.medium.competence;
        wMedPolicy.value = state.weights.medium.policy;
        wMedRob.value = state.weights.medium.robustness;

        wHighComp.value = state.weights.high.competence;
        wHighPolicy.value = state.weights.high.policy;
        wHighRob.value = state.weights.high.robustness;
    }

    function renderConstraintsReference() {
        constraintsReference.innerHTML = "";
        HARD_CONSTRAINTS_REFERENCE.forEach((group) => {
            const header = document.createElement("div");
            header.className = "domain-group-header";
            header.innerText = DOMAIN_LABELS[group.domain] || group.domain;
            constraintsReference.appendChild(header);
            group.items.forEach((item) => {
                const div = document.createElement("div");
                div.className = "constraint-item";
                div.innerHTML = `
                    <span class="badge ${item.badge}">${item.action}</span>
                    <p class="constraint-desc">${item.desc}</p>
                `;
                constraintsReference.appendChild(div);
            });
        });
    }

    // --- EVENTS HANDLERS ---
    function setupEventListeners() {
        tabs.forEach(t => {
            t.addEventListener("click", () => {
                tabs.forEach(btn => btn.classList.remove("active"));
                tabPanes.forEach(pane => pane.classList.remove("active"));

                t.classList.add("active");
                const target = t.dataset.tab;
                state.activeTab = target;
                document.getElementById(`tab-${target}`).classList.add("active");

                const titles = {
                    simulator: ["Gate Simulator", "Evaluate agent planner authorization thresholds in real-time"],
                    benchmarks: ["Safety Benchmarks", "4-way safety mechanism comparison over the real held-out test split"],
                    config: ["Parameters & Rules", "Inspect and tune PEAQ's decision thresholds and hard constraints"],
                };
                pageTitle.innerText = titles[target][0];
                pageSubtitle.innerText = titles[target][1];
            });
        });

        domainSelect.addEventListener("change", async (e) => {
            state.activeDomain = e.target.value;
            await loadTestCases();
            updateAgentPreviews();
        });

        simAgentSelect.addEventListener("change", updateAgentPreviews);

        simTestSelect.addEventListener("change", (e) => {
            const idx = e.target.value;
            if (idx === "") {
                state.selectedCase = null;
                resetSimulatorForm();
                return;
            }
            const test = state.testCases[Number(idx)];
            state.selectedCase = test;
            customRequestGroup.style.display = "none";
            ctxHumanApproved.checked = !!test.context?.human_approved;
            ctxAuthorized.checked = !!test.context?.authorized;
            const rest = { ...test.context };
            delete rest.human_approved;
            delete rest.authorized;
            ctxJsonOverride.value = Object.keys(rest).length ? JSON.stringify(rest, null, 2) : "{}";
        });

        loadCustomBtn.addEventListener("click", () => {
            simTestSelect.value = "";
            state.selectedCase = null;
            customRequestGroup.style.display = "block";
            simCustomText.value = "";
            ctxHumanApproved.checked = false;
            ctxAuthorized.checked = false;
            ctxJsonOverride.value = "{}";
        });

        threshLow.addEventListener("input", (e) => valThreshLow.innerText = e.target.value);
        threshMedium.addEventListener("input", (e) => valThreshMedium.innerText = e.target.value);
        threshHigh.addEventListener("input", (e) => valThreshHigh.innerText = e.target.value);

        saveConfigBtn.addEventListener("click", async () => {
            const payload = {
                thresholds: {
                    low: parseFloat(threshLow.value),
                    medium: parseFloat(threshMedium.value),
                    high: parseFloat(threshHigh.value)
                },
                weights: {
                    low: { competence: parseFloat(wLowComp.value), policy: parseFloat(wLowPolicy.value), robustness: parseFloat(wLowRob.value) },
                    medium: { competence: parseFloat(wMedComp.value), policy: parseFloat(wMedPolicy.value), robustness: parseFloat(wMedRob.value) },
                    high: { competence: parseFloat(wHighComp.value), policy: parseFloat(wHighPolicy.value), robustness: parseFloat(wHighRob.value) }
                }
            };
            try {
                const res = await fetch("/api/config", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.status === "success") {
                    alert("Configuration updated for this server session (not persisted to disk).");
                    state.thresholds = data.thresholds;
                    state.weights = data.weights;
                }
            } catch (e) {
                alert("Error saving configuration: " + e.message);
            }
        });

        runSimulationBtn.addEventListener("click", runSingleSimulation);
        runBenchmarksBtn.addEventListener("click", runBenchmarksSuite);
        runAblationBtn.addEventListener("click", runAblationSuite);
        benchTestFilter.addEventListener("change", filterBenchmarkTable);

        clearConsoleBtn.addEventListener("click", () => {
            consoleOutput.innerHTML = '<span class="console-placeholder">Console cleared.</span>';
        });
    }

    function updateAgentPreviews() {
        const agentId = simAgentSelect.value;
        const profile = state.profiles[agentId];
        if (profile) {
            simAgentPreview.innerHTML = `
                <strong>Metrics:</strong> Competence: ${Math.round(profile.competence*100)}% |
                Policy Adherence: ${Math.round(profile.policy_adherence*100)}% |
                Robustness: ${Math.round(profile.adversarial_robustness*100)}%
                <br><span style="opacity:0.7">${profile.source}</span>
            `;
        }
    }

    function resetSimulatorForm() {
        customRequestGroup.style.display = "none";
        ctxHumanApproved.checked = false;
        ctxAuthorized.checked = false;
        ctxJsonOverride.value = "{}";
    }

    // --- API SIMULATION ---
    async function runSingleSimulation() {
        const agent_id = simAgentSelect.value;
        const config = simConfigSelect.value;
        const use_llm = llmToggle.checked;

        let request_text, action_override, extraNote = "";
        if (state.selectedCase) {
            request_text = state.selectedCase.request_text;
            const actions = state.selectedCase.intended_actions || [];
            action_override = actions[0]?.action;
            if (actions.length > 1) {
                extraNote = ` (case has ${actions.length} total intended actions -- tracer shows decision for action 1 only)`;
            }
        } else {
            const customText = simCustomText.value;
            if (!customText.trim()) {
                alert("Please select a real test case or write a custom user request.");
                return;
            }
            request_text = customText;
            action_override = null;
        }

        let contextOverrides;
        try {
            const raw = ctxJsonOverride.value.trim();
            contextOverrides = raw ? JSON.parse(raw) : {};
        } catch (e) {
            alert("Invalid JSON in 'Additional context overrides': " + e.message);
            return;
        }
        contextOverrides.human_approved = ctxHumanApproved.checked;
        contextOverrides.authorized = ctxAuthorized.checked;

        runSimulationBtn.disabled = true;
        runSimulationBtn.innerHTML = '<i data-lucide="loader" class="animate-spin"></i> <span>Running Gate...</span>';
        lucide.createIcons();

        resetTracerUI();
        consoleOutput.innerHTML = '<span class="console-placeholder">Starting simulation tracer...</span>';
        traceConfigBadge.innerText = simConfigSelect.options[simConfigSelect.selectedIndex].text;

        try {
            const response = await fetch("/api/simulate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    agent_id,
                    request: request_text,
                    domain: state.activeDomain,
                    use_llm,
                    context: contextOverrides,
                    action: action_override,
                })
            });
            const data = await response.json();

            if (data.error) {
                alert("Server Error: " + data.error);
                return;
            }

            renderTracerOutput(data, config, extraNote);
        } catch (e) {
            console.error(e);
            alert("Network Error: " + e.message);
        } finally {
            runSimulationBtn.disabled = false;
            runSimulationBtn.innerHTML = '<i data-lucide="play"></i> <span>Evaluate Safety Gate</span>';
            lucide.createIcons();
        }
    }

    function resetTracerUI() {
        [stepClassify, stepEval, stepConstraints].forEach(el => {
            el.className = "flow-step";
            el.querySelector(".step-output").style.display = "none";
        });
        heroDecision.className = "decision-hero";
        heroDecision.querySelector(".decision-text").innerText = "AWAITING ENGINE";
        heroDecision.querySelector(".decision-reason").innerText = "Trigger evaluation pipeline above.";
    }

    // data is the /api/simulate response: the decide_peaq result plus the classification reason and logs
    function renderTracerOutput(data, config, extraNote) {
        setTimeout(() => {
            stepClassify.classList.add("active", "passed");
            outClassify.innerText = `Action: ${data.action} (Criticality: ${data.criticality.toUpperCase()})${extraNote}\nReason: ${data.classification_reason || "Rules mapping."}`;
            outClassify.style.display = "block";
        }, 300);

        setTimeout(() => {
            stepEval.classList.add("active");
            if (config === "none" || config === "post_hoc") {
                stepEval.classList.add("passed");
                outEval.innerText = `[Bypassed] No pre-planning qualification gate in this configuration.`;
            } else {
                const weights = state.weights[data.criticality];
                const w_str = `Comp: ${weights.competence}*${Math.round(data.scores.competence*100)}% + Policy: ${weights.policy}*${Math.round(data.scores.policy_adherence*100)}% + Rob: ${weights.robustness}*${Math.round(data.scores.adversarial_robustness*100)}%`;
                outEval.innerText = `Trust Score: ${data.trust_score.toFixed(4)} (Threshold: ${data.required_threshold.toFixed(2)}, grey-zone: ${data.grey_zone_margin.toFixed(2)})\nFormula: ${w_str}`;
                if (data.decision === "APPROVE") stepEval.classList.add("passed");
                else if (data.decision === "CONDITIONAL_APPROVE" || data.decision === "HUMAN_REVIEW") stepEval.classList.add("active");
                else stepEval.classList.add("failed");
            }
            outEval.style.display = "block";
        }, 700);

        setTimeout(() => {
            stepConstraints.classList.add("active");
            if (config === "none" || config === "post_hoc") {
                stepConstraints.classList.add("passed");
                outConstraints.innerText = `[Bypassed] Pre-planning hard constraints skipped in this configuration.`;
            } else {
                if (data.hard_constraints_passed) {
                    stepConstraints.classList.add("passed");
                    outConstraints.innerText = "No hard constraint boundaries tripped.";
                } else {
                    stepConstraints.classList.add("failed");
                    outConstraints.innerText = `Tripped (${data.constraint_kind}): ${data.reason}`;
                }
            }
            outConstraints.style.display = "block";
        }, 1100);

        setTimeout(() => {
            let cls = "reject", text = "REJECT";

            if (config === "none") {
                cls = "approve"; text = "DIRECT EXECUTION";
            } else if (config === "post_hoc") {
                cls = "cond-approve"; text = "POST-HOC MONITOR";
            } else if (config === "runtime_auth") {
                if (data.criticality === "high") {
                    cls = data.hard_constraints_passed ? "approve" : "reject";
                    text = data.hard_constraints_passed ? "RUNTIME AUTH: PASSED" : "RUNTIME AUTH: INTERCEPTED";
                } else {
                    cls = "approve"; text = "DIRECT EXECUTION (below runtime-auth tier)";
                }
            } else {
                if (data.decision === "APPROVE") { cls = "approve"; text = "APPROVE (FAST PASS)"; }
                else if (data.decision === "CONDITIONAL_APPROVE") { cls = "cond-approve"; text = "CONDITIONAL_APPROVE"; }
                else if (data.decision === "HUMAN_REVIEW") { cls = "review"; text = "HUMAN_REVIEW"; }
                else { cls = "reject"; text = "REJECT"; }
            }

            heroDecision.className = `decision-hero ${cls}`;
            heroDecision.querySelector(".decision-text").innerText = text;
            heroDecision.querySelector(".decision-reason").innerText = data.reason || "Decided under active configuration parameters.";
        }, 1500);

        setTimeout(() => {
            consoleOutput.innerHTML = "";
            let logIndex = 0;
            const logs = data.logs || [];
            function printNextLine() {
                if (logIndex < logs.length) {
                    const line = logs[logIndex];
                    const div = document.createElement("div");
                    div.className = "console-line";
                    if (line.includes("FAILED") || line.includes("REJECT")) div.classList.add("error");
                    else if (line.includes("PASSED") || line.includes("APPROVE")) div.classList.add("success");
                    else if (line.includes("HUMAN_REVIEW") || line.includes("CONDITIONAL")) div.classList.add("warning");
                    else if (line.includes("SYSTEM")) div.classList.add("system");
                    else div.classList.add("info");
                    div.innerText = `> ${line}`;
                    consoleOutput.appendChild(div);
                    consoleOutput.scrollTop = consoleOutput.scrollHeight;
                    logIndex++;
                    setTimeout(printNextLine, 220);
                }
            }
            printNextLine();
        }, 1800);
    }

    // --- API BENCHMARKS ---
    async function runBenchmarksSuite() {
        const agent_id = benchAgentSelect.value;
        const domain = benchDomainSelect.value;

        runBenchmarksBtn.disabled = true;
        runBenchmarksBtn.innerHTML = '<i data-lucide="loader" class="animate-spin"></i> <span>Compiling Benchmark Metrics...</span>';
        lucide.createIcons();

        benchmarksDashboard.style.display = "none";
        ablationCard.style.display = "none";
        benchmarksPlaceholder.style.display = "block";
        benchmarksPlaceholder.innerHTML = '<i data-lucide="loader" class="placeholder-icon animate-spin"></i><h3>Compiling evaluation matrix...</h3><p>Running 4 safety configurations against the real held-out test split. Please wait...</p>';
        lucide.createIcons();

        try {
            const res = await fetch("/api/benchmark", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ agent_id, domain, split: "test" })
            });
            const data = await res.json();

            if (data.error) {
                alert("Server error: " + data.error);
                benchmarksPlaceholder.innerHTML = '<i data-lucide="play-circle" class="placeholder-icon"></i><h3>No Benchmark Data Compiled</h3><p>' + data.error + '</p>';
                lucide.createIcons();
                return;
            }

            state.activeBenchmarkData = data;
            benchmarksPlaceholder.style.display = "none";
            benchmarksDashboard.style.display = "block";

            updateBenchmarkStats(data);
            renderBenchmarkTable(data);
            renderDetailedTestCasesTable(data);
            drawBenchmarkCharts(data);
        } catch (e) {
            console.error(e);
            alert("Benchmark failed: " + e.message);
        } finally {
            runBenchmarksBtn.disabled = false;
            runBenchmarksBtn.innerHTML = '<i data-lucide="zap"></i> <span>Run 4-Way Configuration Benchmark</span>';
            lucide.createIcons();
        }
    }

    async function runAblationSuite() {
        const agent_id = benchAgentSelect.value;
        const domain = benchDomainSelect.value;

        runAblationBtn.disabled = true;
        runAblationBtn.innerHTML = '<i data-lucide="loader" class="animate-spin"></i> <span>Running Ablation...</span>';
        lucide.createIcons();

        try {
            const res = await fetch("/api/ablation", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ agent_id, domain, split: "test" })
            });
            const data = await res.json();
            if (data.error) {
                alert("Ablation error: " + data.error);
                return;
            }
            renderAblationTable(data.summary);
            ablationCard.style.display = "block";
        } catch (e) {
            console.error(e);
            alert("Ablation failed: " + e.message);
        } finally {
            runAblationBtn.disabled = false;
            runAblationBtn.innerHTML = '<i data-lucide="layers-3"></i> <span>Run Hard-Constraint / Trust-Score Ablation</span>';
            lucide.createIcons();
        }
    }

    function renderAblationTable(summary) {
        ablationTable.innerHTML = "";
        const labels = {
            hard_constraint_only: "Hard-Constraint Only",
            trust_score_only: "Trust-Score Only",
            combined: "Combined (Real PEAQ Gate)",
        };
        Object.entries(labels).forEach(([key, label]) => {
            const s = summary[key];
            if (!s) return;
            const tr = document.createElement("tr");
            if (key === "combined") { tr.style.fontWeight = "bold"; tr.style.backgroundColor = "rgba(99, 102, 241, 0.05)"; }
            tr.innerHTML = `
                <td><strong>${label}</strong></td>
                <td class="text-green">${s.catch_rate}%</td>
                <td class="${s.false_rejection_rate > 50 ? 'text-red' : 'text-amber'}">${s.false_rejection_rate}%</td>
                <td>${s.bad_cases_blocked} / ${s.bad_cases}</td>
            `;
            ablationTable.appendChild(tr);
        });
    }

    function updateBenchmarkStats(data) {
        const bgNone = data.benchmarks["none"];
        const bgPeaq = data.benchmarks["peaq_runtime"];
        const bgRuntime = data.benchmarks["runtime_auth"];

        statViolations.innerText = Math.max(0, bgNone.policy_violations - bgPeaq.policy_violations);
        statBypasses.innerText = Math.max(0, bgNone.adversarial_bypasses - bgPeaq.adversarial_bypasses);

        const rtLat = bgRuntime.avg_latency;
        const peaqLat = bgPeaq.avg_latency;
        let latencySavings = 0;
        if (rtLat > 0) latencySavings = Math.round(((rtLat - peaqLat) / rtLat) * 100);
        statEfficiency.innerText = latencySavings > 0 ? `${latencySavings}%` : `0%`;
    }

    function renderBenchmarkTable(data) {
        benchComparisonTable.innerHTML = "";
        const configs = ["none", "post_hoc", "runtime_auth", "peaq_runtime"];
        const labels = {
            none: "No Safety Gate",
            post_hoc: "Post-hoc Checking",
            runtime_auth: "Runtime Auth Only",
            peaq_runtime: "PEAQ Gate (Proposed)"
        };

        configs.forEach(cfg => {
            const b = data.benchmarks[cfg];
            const tr = document.createElement("tr");
            if (cfg === "peaq_runtime") {
                tr.style.fontWeight = "bold";
                tr.style.backgroundColor = "rgba(99, 102, 241, 0.05)";
            }
            const catchStr = b.catch_rate === null ? "N/A" : `${b.catch_rate}%`;
            const frStr = b.false_rejection_rate === null ? "N/A" : `${b.false_rejection_rate}%`;
            tr.innerHTML = `
                <td><strong>${labels[cfg]}</strong></td>
                <td class="text-green">${catchStr}</td>
                <td class="${b.false_rejection_rate > 50 ? 'text-red' : 'text-amber'}">${frStr}</td>
                <td class="${b.policy_violations > 0 ? 'text-red' : 'text-green'}">${b.policy_violations}</td>
                <td class="${b.adversarial_bypasses > 0 ? 'text-red' : 'text-green'}">${b.adversarial_bypasses}</td>
                <td><strong>${b.avg_latency}s</strong></td>
            `;
            benchComparisonTable.appendChild(tr);
        });
    }

    function renderDetailedTestCasesTable(data) {
        state.peaqCaseResults = data.benchmarks["peaq_runtime"].case_results || [];
        filterBenchmarkTable();
    }

    function outcomeBadge(r) {
        if (r.policy_violated || r.adversarial_bypass) return '<span class="badge badge-red">Violation Slipped Through</span>';
        if (r.false_rejection) return '<span class="badge badge-orange">False Rejection</span>';
        if (r.false_human_review) return '<span class="badge badge-orange">False Human Review</span>';
        if (r.ground_truth_label !== "benign" && r.blocked_bad_action) return '<span class="badge badge-green">Correctly Blocked</span>';
        return '<span class="badge badge-green">Correctly Approved</span>';
    }

    function filterBenchmarkTable() {
        if (!state.peaqCaseResults) return;
        const filter = benchTestFilter.value;
        benchDetailTable.innerHTML = "";

        state.peaqCaseResults.forEach(r => {
            if (filter === "benign" && r.ground_truth_label !== "benign") return;
            if (filter === "bad" && r.ground_truth_label === "benign") return;
            if (filter === "violations" && !(r.policy_violated || r.adversarial_bypass)) return;
            if (filter === "false_rejections" && !r.false_rejection) return;

            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><code>${r.case_id}</code></td>
                <td>${r.source}</td>
                <td>${r.domain}</td>
                <td><span class="badge">${r.ground_truth_label.toUpperCase()}</span></td>
                <td>${r.total_latency}s</td>
                <td>${outcomeBadge(r)}</td>
            `;
            benchDetailTable.appendChild(tr);
        });
    }

    function drawBenchmarkCharts(data) {
        const configs = ["none", "post_hoc", "runtime_auth", "peaq_runtime"];
        const labels = ["No Gate", "Post-hoc", "Runtime Auth", "PEAQ Gate"];

        const catchData = configs.map(c => data.benchmarks[c].catch_rate ?? 0);
        const frData = configs.map(c => data.benchmarks[c].false_rejection_rate ?? 0);
        const latencyData = configs.map(c => data.benchmarks[c].avg_latency);

        if (state.chartSafety) state.chartSafety.destroy();
        const ctxSafety = document.getElementById("chart-safety").getContext("2d");
        state.chartSafety = new Chart(ctxSafety, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    { label: "Catch Rate (%)", data: catchData, backgroundColor: "rgba(16, 185, 129, 0.7)", borderColor: "rgba(16, 185, 129, 1)", borderWidth: 1 },
                    { label: "False-Rejection Rate (%)", data: frData, backgroundColor: "rgba(239, 68, 68, 0.7)", borderColor: "rgba(239, 68, 68, 1)", borderWidth: 1 }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, max: 100, grid: { color: "#23273a" }, ticks: { color: "#9ca3af", callback: v => `${v}%` } },
                    x: { grid: { display: false }, ticks: { color: "#9ca3af" } }
                },
                plugins: { legend: { labels: { color: "#f3f4f6", font: { family: "Inter" } } } }
            }
        });

        if (state.chartLatency) state.chartLatency.destroy();
        const ctxLatency = document.getElementById("chart-latency").getContext("2d");
        state.chartLatency = new Chart(ctxLatency, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{ label: "Avg Total Latency (seconds)", data: latencyData, backgroundColor: "rgba(99, 102, 241, 0.6)", borderColor: "rgba(99, 102, 241, 1)", borderWidth: 1 }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, grid: { color: "#23273a" }, ticks: { color: "#9ca3af", callback: v => `${v}s` } },
                    x: { grid: { display: false }, ticks: { color: "#9ca3af" } }
                },
                plugins: { legend: { labels: { color: "#f3f4f6", font: { family: "Inter" } } } }
            }
        });
    }

    init();
});
