

 function Navbar() {
  return (
    <nav className="fixed top-0 w-full z-50 bg-black/80 backdrop-blur border-b border-white/5">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-2 text-white font-semibold">
          <span className="w-3 h-3 rounded-sm bg-emerald-500" />
          Quant Research Platform
        </div>

        <div className="flex items-center gap-4">
          <button className="text-sm text-gray-300 hover:text-white">
            Login
          </button>
          <button className="px-4 py-2 rounded-md bg-emerald-500 text-black text-sm font-medium hover:bg-emerald-400">
            Sign Up
          </button>
        </div>
      </div>
    </nav>
  );
}

export default Navbar
