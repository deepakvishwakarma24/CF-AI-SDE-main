// "use client";

// import { useEffect, useState } from "react";
// import { motion } from "framer-motion";
// import Link from "next/link";

// type Article = {
//   title: string;
//   source: { name: string };
//   url: string;
//   publishedAt: string;
// };

// export default function TopStoriesToday() {
//   const [articles, setArticles] = useState<Article[]>([]);
//   const [loading, setLoading] = useState(true);

//   useEffect(() => {
//     fetch("/api/top-stories")
//       .then(res => res.json())
//       .then(data => {
//         setArticles(data);
//         setLoading(false);
//       });
//   }, []);

//   return (
//     <section className="py-24 bg-black">
//       <div className="max-w-7xl mx-auto px-6">
//         <h2 className="text-2xl font-semibold text-white mb-6">
//           Top Stories Today
//         </h2>

//         {loading ? (
//           <p className="text-gray-500">Loading latest insights...</p>
//         ) : (
//         //   <div className="grid md:grid-cols-3 gap-6">
//         //     {articles.map((a, i) => (
//         //       <a
//         //         key={i}
//         //         href={a.url}
//         //         target="_blank"
//         //         rel="noopener noreferrer"
//         //         className="rounded-xl border border-white/10 bg-zinc-900/60 p-5 hover:border-emerald-400/30 transition"
//         //       >
//         //         <p className="text-sm text-gray-400 mb-2">
//         //           {a.source.name}
//         //         </p>

//         //         <h3 className="text-white font-medium leading-snug">
//         //           {a.title}
//         //         </h3>

//         //         <p className="mt-3 text-xs text-gray-500">
//         //           {new Date(a.publishedAt).toLocaleDateString()}
//         //         </p>
//         //       </a>
//         //     ))}
//         //   </div>


//         <motion.div
//   key={article.url}
//   initial={{ opacity: 0, y: 12 }}
//   animate={{ opacity: 1, y: 0 }}
//   transition={{ duration: 0.4, ease: "easeOut" }}
//   whileHover={{
//     y: -6,
//     boxShadow: "0 0 0 1px rgba(16,185,129,0.25)",
//   }}
//   className="group relative rounded-xl border border-white/10 bg-gradient-to-br from-zinc-900/60 to-black/60 p-5 backdrop-blur"
// >
//   {/* SOURCE */}
//   <p className="text-xs uppercase tracking-widest text-emerald-400/80">
//     {article.source?.name}
//   </p>

//   {/* TITLE */}
//   <h3 className="mt-2 text-sm font-medium text-white leading-snug">
//     {article.title}
//   </h3>

//   {/* DATE */}
//   <p className="mt-2 text-xs text-gray-500">
//     {new Date(article.publishedAt).toLocaleDateString()}
//   </p>

//   {/* READ LINK */}
//   <Link
//     href={article.url}
//     target="_blank"
//     className="mt-4 inline-flex items-center gap-1 text-xs text-emerald-400 opacity-0 group-hover:opacity-100 transition"
//   >
//     Read full story →
//   </Link>

//   {/* SUBTLE LIVE GLOW */}
//   <span className="pointer-events-none absolute inset-0 rounded-xl opacity-0 group-hover:opacity-100 transition">
//     <span className="absolute inset-0 rounded-xl bg-emerald-500/5 blur-xl" />
//   </span>
// </motion.div>
//         )}
//       </div>
//     </section>
//   );
// }



"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";

type Article = {
  title: string;
  source: { name: string };
  url: string;
  publishedAt: string;
};

export default function TopStoriesToday() {
  const [articles, setArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/top-stories")
      .then(res => res.json())
      .then(data => {
        setArticles(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  return (
    <section className="py-24 bg-black">
      <div className="max-w-7xl mx-auto px-6">
        {/* SECTION TITLE */}
        <h2 className="text-2xl font-semibold text-white mb-8">
          Top Stories Today
        </h2>

        {/* LOADING STATE */}
        {loading ? (
          <p className="text-gray-500 animate-pulse">
            Loading latest insights…
          </p>
        ) : (
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {articles.map((article, i) => (
              <motion.div
                key={article.url || i}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{
                  duration: 0.4,
                  ease: "easeOut",
                  delay: i * 0.05,
                }}
                whileHover={{
                  y: -6,
                  boxShadow: "0 0 0 1px rgba(16,185,129,0.25)",
                }}
                className="group relative rounded-xl border border-white/10
                           bg-gradient-to-br from-zinc-900/60 to-black/60
                           p-5 backdrop-blur"
              >
                {/* SOURCE */}
                <p className="text-xs uppercase tracking-widest text-emerald-400/80">
                  {article.source?.name || "Market News"}
                </p>

                {/* TITLE */}
                <h3 className="mt-2 text-sm font-medium text-white leading-snug">
                  {article.title}
                </h3>

                {/* DATE */}
                <p className="mt-2 text-xs text-gray-500">
                  {new Date(article.publishedAt).toLocaleDateString()}
                </p>

                {/* READ LINK */}
                <Link
                  href={article.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-4 inline-flex items-center gap-1
                             text-xs text-emerald-400
                             opacity-0 group-hover:opacity-100
                             transition"
                >
                  Read full story →
                </Link>

                {/* SUBTLE LIVE GLOW */}
                <span className="pointer-events-none absolute inset-0 rounded-xl
                                 opacity-0 group-hover:opacity-100 transition">
                  <span className="absolute inset-0 rounded-xl
                                   bg-emerald-500/5 blur-xl" />
                </span>
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

