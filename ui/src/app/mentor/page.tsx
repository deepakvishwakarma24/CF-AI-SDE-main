// import React from 'react'

// function MentorPage() {
//   return (
//     <div>
//       <h1 className="text-2xl font-semibold mb-2">Trading Mentor</h1>
//       <p className="text-gray-400">
//         Ask the AI why your strategy worked or failed.
//       </p>
//     </div>
//   );
// }

// export default MentorPage


"use client";

import { useState } from "react";
import { mockMentorResponse } from "@/data/mentorResponse";
import ClientShell from "@/components/ClientShell";

 function MentorPage() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState<string | null>(null);
  const [isThinking, setIsThinking] = useState(false);

  const askMentor = () => {
    if (!question) return;

    setIsThinking(true);
    setResponse(null);

    setTimeout(() => {
      setResponse(mockMentorResponse);
      setIsThinking(false);
    }, 1500);
  };

  return (
    <ClientShell>
    <div className="max-w-3xl mx-auto space-y-6">
      <h1 className="text-2xl font-semibold">Trading Mentor</h1>

      {/* Question Input */}
      <div className="flex gap-2">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Why did my strategy underperform in March 2020?"
          className="flex-1 bg-gray-900 border border-gray-700 rounded px-3 py-2"
        />
        <button
          onClick={askMentor}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded"
        >
          Ask
        </button>
      </div>

      {/* Thinking */}
      {isThinking && (
        <div className="text-gray-400">Mentor is analyzing...</div>
      )}

      {/* Response */}
      {response && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 prose prose-invert max-w-none">
          <div dangerouslySetInnerHTML={{ __html: markdownToHtml(response) }} />
        </div>
      )}
    </div>
    </ClientShell>
  );
}

/* Simple markdown renderer (minimal) */
function markdownToHtml(markdown: string) {
  return markdown
    .replace(/### (.*)/g, "<h3>$1</h3>")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br/>");
}


export default MentorPage

