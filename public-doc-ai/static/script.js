// Confirm JavaScript is connected
console.log("✅ script.js is connected and running.");

// ──────────────────────────────────────────────────────────────────
// DOM REFERENCES
// ──────────────────────────────────────────────────────────────────
const textInput = document.getElementById("text-input");
const pdfInput = document.getElementById("pdf-input");
const dropZone = document.getElementById("drop-zone");
const fileNameLabel = document.getElementById("file-name");
const analyzeBtn = document.getElementById("analyze-btn");
const loadingCard = document.getElementById("loading-card");
const loadingMessage = document.getElementById("loading-message");
const resultCard = document.getElementById("result-card");
const resultTitle = document.getElementById("result-title");
const resultTypeBadge = document.getElementById("result-type-badge");
const resultContent = document.getElementById("result");
const errorBox = document.getElementById("error-box");
const suggestedSection = document.getElementById("suggested-section");
const suggestedQuestions = document.getElementById("suggested-questions");

// Q&A elements
const qaSection = document.getElementById("qa-section");
const questionInput = document.getElementById("question-input");
const askBtn = document.getElementById("ask-btn");
const qaLoading = document.getElementById("qa-loading");
const chatHistory = document.getElementById("chat-history");

// ──────────────────────────────────────────────────────────────────
// LOADING MESSAGES — Cycle through these while waiting
// ──────────────────────────────────────────────────────────────────
const LOADING_MESSAGES = [
    "Extracting document...",
    "Identifying document type...",
    "Analyzing key details...",
    "Extracting entities & facts...",
    "Generating structured report...",
    "Almost done..."
];

let loadingInterval = null;

function startLoadingMessages() {
    let index = 0;
    loadingMessage.textContent = LOADING_MESSAGES[0];

    loadingInterval = setInterval(function () {
        index++;
        if (index < LOADING_MESSAGES.length) {
            loadingMessage.style.opacity = "0";
            setTimeout(function () {
                loadingMessage.textContent = LOADING_MESSAGES[index];
                loadingMessage.style.opacity = "1";
            }, 200);
        }
    }, 2500);
}

function stopLoadingMessages() {
    if (loadingInterval) {
        clearInterval(loadingInterval);
        loadingInterval = null;
    }
}

// ──────────────────────────────────────────────────────────────────
// DRAG & DROP + FILE PICKER
// ──────────────────────────────────────────────────────────────────

// Click on drop zone opens file picker
dropZone.addEventListener("click", function () {
    pdfInput.click();
});

// File selected via picker
pdfInput.addEventListener("change", function () {
    handleFileSelection(pdfInput.files);
});

// Drag events
dropZone.addEventListener("dragover", function (e) {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", function () {
    dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", function (e) {
    e.preventDefault();
    dropZone.classList.remove("drag-over");

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        pdfInput.files = files;
        handleFileSelection(files);
    }
});

function handleFileSelection(files) {
    if (files.length > 0) {
        const name = files[0].name;
        fileNameLabel.textContent = "📎 " + name;
        fileNameLabel.classList.add("active");
        dropZone.classList.add("has-file");
    } else {
        fileNameLabel.textContent = "No file selected";
        fileNameLabel.classList.remove("active");
        dropZone.classList.remove("has-file");
    }
}

// ──────────────────────────────────────────────────────────────────
// MARKDOWN RENDERING — Full support for AI responses
// ──────────────────────────────────────────────────────────────────

// Section icon mapping — covers the new prompt output format
const SECTION_ICONS = {
    "overview":                    "📝",
    "key highlights":              "🔑",
    "important people":            "👤",
    "organizations":               "🏢",
    "important dates":             "📅",
    "deadlines":                   "📅",
    "financial":                   "💰",
    "risks":                       "⚠️",
    "warnings":                    "⚠️",
    "action items":                "✅",
    "next steps":                  "✅",
    "final takeaway":              "💡",
    "executive summary":           "📋",
    "summary":                     "💡",
    "eligibility":                 "✅",
    "documents required":          "📄",
    "what is this about":          "📋",
    "who is this for":             "👥",
    "experience":                  "💼",
    "skills":                      "🛠️",
    "education":                   "🎓",
    "methodology":                 "🔬",
    "results":                     "📊",
    "conclusion":                  "🏁",
    "found in":                    "📍",
    "confidence":                  "🎯"
};

function getIconForHeading(headingText) {
    const lower = headingText.toLowerCase();
    for (const [keyword, icon] of Object.entries(SECTION_ICONS)) {
        if (lower.includes(keyword)) return icon;
    }
    return "📌";
}

function renderMarkdown(text) {
    const lines = text.split("\n");
    let html = "";
    let inList = false;
    let listType = "";
    let inTable = false;
    let tableRows = [];

    function closeList() {
        if (inList) {
            html += listType === "ul" ? "</ul>" : "</ol>";
            inList = false;
        }
    }

    function closeTable() {
        if (inTable && tableRows.length > 0) {
            html += '<div class="table-wrap"><table>';
            for (let r = 0; r < tableRows.length; r++) {
                const cells = tableRows[r];
                const tag = r === 0 ? "th" : "td";
                html += "<tr>";
                for (const cell of cells) {
                    html += "<" + tag + ">" + renderInline(cell.trim()) + "</" + tag + ">";
                }
                html += "</tr>";
            }
            html += "</table></div>";
            tableRows = [];
            inTable = false;
        }
    }

    for (let i = 0; i < lines.length; i++) {
        let line = lines[i];

        // ## Headings with optional emoji prefix
        const h2Match = line.match(/^##\s+(.+)$/);
        if (h2Match) {
            closeList();
            closeTable();
            let headingText = h2Match[1].trim();
            // Strip leading emoji if present (the heading text already has one from the prompt)
            const emojiMatch = headingText.match(/^([\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE00}-\u{FEFF}]\uFE0F?\s*)/u);
            let icon = "";
            let cleanText = headingText;
            if (emojiMatch) {
                icon = emojiMatch[1].trim();
                cleanText = headingText.slice(emojiMatch[0].length).trim();
            } else {
                icon = getIconForHeading(headingText);
            }
            html += '<div class="section-heading"><span class="section-icon">' + icon + '</span> ' + renderInline(cleanText) + '</div>';
            continue;
        }

        // Table rows: | col | col | col |
        const tableMatch = line.match(/^\|(.+)\|$/);
        if (tableMatch) {
            closeList();
            // Check if it's a separator row like |---|---|
            const content = tableMatch[1];
            if (/^[\s\-|:]+$/.test(content)) {
                // Separator row — skip but mark table as started
                inTable = true;
                continue;
            }
            inTable = true;
            const cells = content.split("|").map(c => c.trim());
            tableRows.push(cells);
            continue;
        } else if (inTable) {
            closeTable();
        }

        // Bold headings: **Something** on its own line (legacy format)
        const boldHeadingMatch = line.match(/^\*\*(\d+\.\s*)?(.+?)\*\*\s*$/);
        if (boldHeadingMatch && !line.match(/^\s*[-*]\s/)) {
            closeList();
            const headingText = boldHeadingMatch[2];
            const icon = getIconForHeading(headingText);
            html += '<div class="section-heading"><span class="section-icon">' + icon + '</span> ' + renderInline(headingText) + '</div>';
            continue;
        }

        // Bullet list items: - text or * text
        const bulletMatch = line.match(/^\s*[-*]\s+(.+)$/);
        if (bulletMatch) {
            closeTable();
            if (!inList || listType !== "ul") {
                closeList();
                html += "<ul>";
                inList = true;
                listType = "ul";
            }
            html += "<li>" + renderInline(bulletMatch[1]) + "</li>";
            continue;
        }

        // Numbered list items: 1. text
        const numberedMatch = line.match(/^\s*\d+\.\s+(.+)$/);
        if (numberedMatch) {
            closeTable();
            if (!inList || listType !== "ol") {
                closeList();
                html += "<ol>";
                inList = true;
                listType = "ol";
            }
            html += "<li>" + renderInline(numberedMatch[1]) + "</li>";
            continue;
        }

        // Close any open list for other content
        closeList();

        // Empty line
        if (line.trim() === "") continue;

        // Regular paragraph
        html += "<p>" + renderInline(line) + "</p>";
    }

    closeList();
    closeTable();
    return html;
}

function renderInline(text) {
    // Bold: **text**
    text = text.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    text = text.replace(/__(.+?)__/g, "<strong>$1</strong>");
    // Italic: *text* (not inside bold)
    text = text.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, "<em>$1</em>");
    // Inline code: `text`
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    return text;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// ──────────────────────────────────────────────────────────────────
// SUGGESTED QUESTIONS — Render clickable chips
// ──────────────────────────────────────────────────────────────────

function renderSuggestedQuestions(questions) {
    suggestedQuestions.innerHTML = "";
    suggestedSection.classList.add("hidden");

    if (!questions || questions.length === 0) return;

    questions.forEach(function (q) {
        const chip = document.createElement("button");
        chip.className = "suggested-chip";
        chip.textContent = q;
        chip.addEventListener("click", function () {
            questionInput.value = q;
            questionInput.focus();
            // Auto-submit
            askQuestion();
        });
        suggestedQuestions.appendChild(chip);
    });

    suggestedSection.classList.remove("hidden");
}

// ──────────────────────────────────────────────────────────────────
// ANALYZE BUTTON
// ──────────────────────────────────────────────────────────────────
analyzeBtn.addEventListener("click", async function () {
    const text = textInput.value.trim();
    const hasFile = pdfInput.files.length > 0;

    if (!text && !hasFile) {
        showError("⚠️ Please paste some text or upload a PDF.");
        return;
    }

    const formData = new FormData();
    if (hasFile) formData.append("pdf", pdfInput.files[0]);
    if (text) formData.append("text", text);

    // Show loading, hide previous results
    loadingCard.classList.remove("hidden");
    resultCard.classList.add("hidden");
    errorBox.classList.add("hidden");
    qaSection.classList.add("hidden");
    suggestedSection.classList.add("hidden");
    chatHistory.innerHTML = "";
    analyzeBtn.disabled = true;
    startLoadingMessages();

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!data.success) {
            showError("❌ " + data.error);
        } else {
            // Update result card header with dynamic title and type
            if (data.doc_title) {
                resultTitle.textContent = "📄 " + data.doc_title;
            } else {
                resultTitle.textContent = "📄 AI Analysis Report";
            }

            if (data.doc_type) {
                resultTypeBadge.textContent = data.doc_type;
                resultTypeBadge.classList.remove("hidden");
            } else {
                resultTypeBadge.classList.add("hidden");
            }

            // Render analysis with full markdown support
            resultContent.innerHTML = renderMarkdown(data.analysis);
            resultCard.classList.remove("hidden");

            // Render suggested questions
            renderSuggestedQuestions(data.suggested_questions);

            // Show Q&A section
            qaSection.classList.remove("hidden");
            questionInput.value = "";

            // Scroll to result
            resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
        }
    } catch (err) {
        showError("❌ Something went wrong. Is the server running?");
        console.error("Fetch error:", err);
    } finally {
        stopLoadingMessages();
        loadingCard.classList.add("hidden");
        analyzeBtn.disabled = false;
    }
});

function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
}

// ──────────────────────────────────────────────────────────────────
// ASK QUESTION — Now renders markdown in answers
// ──────────────────────────────────────────────────────────────────
async function askQuestion() {
    const question = questionInput.value.trim();
    if (!question) return;

    // Show loading
    qaLoading.classList.remove("hidden");
    askBtn.disabled = true;

    // Add user bubble
    const questionDiv = document.createElement("div");
    questionDiv.className = "chat-bubble user-bubble";
    questionDiv.innerHTML = '<div class="user-label">You</div>' + escapeHtml(question);
    chatHistory.appendChild(questionDiv);

    questionInput.value = "";

    try {
        const response = await fetch("/api/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: question })
        });

        const data = await response.json();
        const answerDiv = document.createElement("div");

        if (!data.success) {
            answerDiv.className = "chat-bubble error-bubble";
            answerDiv.textContent = "❌ " + data.error;
        } else {
            answerDiv.className = "chat-bubble answer-bubble";
            // Render markdown in Q&A answers too (not just escapeHtml)
            answerDiv.innerHTML =
                '<div class="answer-label">AI</div>' +
                '<div class="answer-text">' + renderMarkdown(data.answer) + '</div>';
        }

        chatHistory.appendChild(answerDiv);
        answerDiv.scrollIntoView({ behavior: "smooth", block: "end" });

    } catch (err) {
        const errorDiv = document.createElement("div");
        errorDiv.className = "chat-bubble error-bubble";
        errorDiv.textContent = "❌ Something went wrong. Is the server running?";
        chatHistory.appendChild(errorDiv);
        console.error("Fetch error:", err);
    } finally {
        qaLoading.classList.add("hidden");
        askBtn.disabled = false;
        questionInput.focus();
    }
}

// Click + Enter key
askBtn.addEventListener("click", askQuestion);
questionInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter") askQuestion();
});
