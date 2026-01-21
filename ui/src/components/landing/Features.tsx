const features = [
  { title: "Strategy Builder", desc: "Define rule-based strategies using indicators and risk controls." },
  { title: "Backtesting Engine", desc: "Evaluate strategy performance on historical data." },
  { title: "Indicator Research Lab", desc: "Experiment with technical indicators visually." },
  { title: "AI Trading Mentor", desc: "AI-driven explanations of strategy behavior." },
];

 function Features() {
  return (
    <section className="bg-black py-24">
      <div className="max-w-7xl mx-auto px-6">
        <h2 className="text-center text-white text-3xl font-semibold mb-12">
          Engineered for Precision
        </h2>

        <div className="grid md:grid-cols-4 gap-6">
          {features.map((f) => (
            <div key={f.title} className="p-6 border border-white/10 rounded-xl bg-black hover:border-emerald-400/40 transition">
              <h3 className="text-white font-medium mb-2">{f.title}</h3>
              <p className="text-sm text-gray-400">{f.desc}</p>
              <span className="mt-4 inline-block text-xs text-emerald-400">
                Explore Module →
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}


export default Features