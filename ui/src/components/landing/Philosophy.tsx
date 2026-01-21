import BackgroundTrendChart from "../visuals/BackgroundTrendChart";
import CandlestickInfographic from "../visuals/CandlestickInfographic";

 function Philosophy() {
  return (
    <section className="relative py-32 overflow-hidden">
       <CandlestickInfographic/>
{/* 
      <div className="max-w-7xl mx-auto px-6 grid md:grid-cols-2 gap-12"> */}
      <div className="relative z-10 max-w-7xl mx-auto px-6 grid md:grid-cols-2 gap-12">
        <div>
          <span className="text-emerald-400 text-xs tracking-widest">
            OUR METHODOLOGY
          </span>
          <h2 className="mt-4 text-4xl font-semibold text-white">
            Built for Research,
            <br />
            <span className="text-gray-400 italic">Not Guesswork</span>
          </h2>
          <p className="mt-6 text-gray-400">
            We remove emotional bias and replace it with systematic,
            data-driven confidence.
          </p>
        </div>

        <ul className="space-y-4">
          {[
            "Declarative strategy definitions",
            "Backend-driven computation",
            "ML-ready, API-first architecture",
            "Clear separation of UI, logic, and intelligence",
          ].map(item => (
            <li key={item} className="p-4 border border-white/10 rounded-lg text-gray-300">
              ● {item}
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}



export default Philosophy