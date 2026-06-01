import { Header } from './components/Layout/Header'
import { Sidebar } from './components/Layout/Sidebar'
import { MainContent } from './components/Layout/MainContent'

function App() {
  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <MainContent />
      </div>
    </div>
  )
}

export default App
