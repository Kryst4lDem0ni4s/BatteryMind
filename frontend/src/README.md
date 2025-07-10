frontend/
├── public/
│   ├── index.html
│   ├── favicon.ico
│   ├── manifest.json
│   └── robots.txt
├── src/
│   ├── index.tsx
│   ├── App.tsx
│   ├── App.css
│   ├── components/
│   │   ├── common/
│   │   │   ├── Header/
│   │   │   │   ├── Header.tsx
│   │   │   │   ├── Header.styles.ts
│   │   │   │   └── index.ts
│   │   │   ├── Sidebar/
│   │   │   │   ├── Sidebar.tsx
│   │   │   │   ├── Sidebar.styles.ts
│   │   │   │   └── index.ts
│   │   │   ├── LoadingSpinner/
│   │   │   ├── ErrorBoundary/
│   │   │   ├── Modal/
│   │   │   └── Notification/
│   │   ├── dashboard/
│   │   │   ├── DashboardOverview/
│   │   │   ├── BatteryHealthChart/
│   │   │   ├── FleetStatus/
│   │   │   ├── RealTimeMetrics/
│   │   │   ├── AlertsPanel/
│   │   │   └── AutonomousDecisions/
│   │   ├── battery/
│   │   │   ├── BatteryCard/
│   │   │   ├── BatteryDetails/
│   │   │   ├── BatteryHistory/
│   │   │   ├── BatteryPrediction/
│   │   │   └── BatteryPassport/
│   │   ├── fleet/
│   │   │   ├── FleetManagement/
│   │   │   ├── FleetAnalytics/
│   │   │   ├── FleetOptimization/
│   │   │   └── FleetComparison/
│   │   ├── ai/
│   │   │   ├── AIInsights/
│   │   │   ├── PredictiveAnalytics/
│   │   │   ├── AutonomousAgents/
│   │   │   ├── FederatedLearning/
│   │   │   └── ModelPerformance/
│   │   ├── blockchain/
│   │   │   ├── BlockchainExplorer/
│   │   │   ├── TransactionHistory/
│   │   │   ├── QRCodeScanner/
│   │   │   └── BatteryPassport/
│   │   └── circular-economy/
│   │       ├── CircularDashboard/
│   │       ├── RecyclingOptimization/
│   │       ├── SecondaryMarket/
│   │       └── SustainabilityMetrics/
│   ├── pages/
│   │   ├── Dashboard/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Dashboard.styles.ts
│   │   │   └── index.ts
│   │   ├── BatteryManagement/
│   │   ├── FleetOverview/
│   │   ├── AIInsights/
│   │   ├── BlockchainTracker/
│   │   ├── CircularEconomy/
│   │   ├── Settings/
│   │   └── Profile/
│   ├── hooks/
│   │   ├── useAuth.ts
│   │   ├── useBatteryData.ts
│   │   ├── useWebSocket.ts
│   │   ├── useRealTimeData.ts
│   │   ├── useNotifications.ts
│   │   └── useLocalStorage.ts
│   ├── services/
│   │   ├── api/
│   │   │   ├── batteryApi.ts
│   │   │   ├── fleetApi.ts
│   │   │   ├── predictionApi.ts
│   │   │   ├── blockchainApi.ts
│   │   │   ├── authApi.ts
│   │   │   └── index.ts
│   │   ├── websocket/
│   │   │   ├── websocketClient.ts
│   │   │   ├── websocketTypes.ts
│   │   │   └── index.ts
│   │   ├── blockchain/
│   │   │   ├── web3Service.ts
│   │   │   ├── contractService.ts
│   │   │   └── index.ts
│   │   └── utils/
│   │       ├── httpClient.ts
│   │       ├── errorHandler.ts
│   │       ├── dataProcessor.ts
│   │       └── validators.ts
│   ├── store/
│   │   ├── index.ts
│   │   ├── store.ts
│   │   ├── slices/
│   │   │   ├── authSlice.ts
│   │   │   ├── batterySlice.ts
│   │   │   ├── fleetSlice.ts
│   │   │   ├── predictionSlice.ts
│   │   │   ├── blockchainSlice.ts
│   │   │   ├── notificationSlice.ts
│   │   │   └── settingsSlice.ts
│   │   └── middleware/
│   │       ├── apiMiddleware.ts
│   │       ├── loggerMiddleware.ts
│   │       └── persistenceMiddleware.ts
│   ├── types/
│   │   ├── battery.ts
│   │   ├── fleet.ts
│   │   ├── prediction.ts
│   │   ├── blockchain.ts
│   │   ├── user.ts
│   │   ├── api.ts
│   │   └── index.ts
│   ├── utils/
│   │   ├── formatters.ts
│   │   ├── validators.ts
│   │   ├── constants.ts
│   │   ├── helpers.ts
│   │   ├── dateUtils.ts
│   │   └── chartUtils.ts
│   ├── styles/
│   │   ├── globals.css
│   │   ├── variables.css
│   │   ├── mixins.css
│   │   ├── reset.css
│   │   └── components.css
│   ├── assets/
│   │   ├── images/
│   │   ├── icons/
│   │   ├── fonts/
│   │   └── animations/
│   ├── contexts/
│   │   ├── AuthContext.tsx
│   │   ├── ThemeContext.tsx
│   │   ├── WebSocketContext.tsx
│   │   └── NotificationContext.tsx
│   └── config/
│       ├── constants.ts
│       ├── apiConfig.ts
│       ├── chartConfig.ts
│       └── themeConfig.ts
├── tests/
│   ├── __mocks__/
│   ├── setup.ts
│   ├── components/
│   ├── pages/
│   ├── hooks/
│   ├── services/
│   ├── store/
│   └── utils/
├── package.json
├── tsconfig.json
├── tailwind.config.js
├── jest.config.js
├── cypress.config.ts
├── .eslintrc.js
├── .prettierrc
├── Dockerfile
└── .env.example
