{
  "config_info": {
    "name": "EA GlobalFlow Pro Risk Management Configuration",
    "version": "v0.1",
    "created_date": "2025-01-01",
    "description": "Comprehensive risk management settings with VIX-based position sizing",
    "criticality": "HIGHEST",
    "owner": "Risk Management Team"
  },
  
  "risk_framework": {
    "description": "Core risk management framework settings",
    "enabled": true,
    "framework_type": "VIX_ADAPTIVE_RISK",
    "base_currency": "INR",
    "account_type": "INDIVIDUAL_TRADING",
    
    "core_principles": {
      "capital_preservation": true,
      "risk_adjusted_returns": true,
      "dynamic_position_sizing": true,
      "correlation_management": true,
      "volatility_adjustment": true,
      "emergency_protocols": true
    }
  },
  
  "position_sizing": {
    "description": "VIX-based dynamic position sizing configuration",
    
    "base_settings": {
      "method": "vix_adaptive",
      "base_risk_percent": 1.0,
      "max_risk_per_trade": 2.0,
      "min_risk_per_trade": 0.25,
      "account_heat_limit": 20.0,
      "position_correlation_limit": 0.7
    },
    
    "vix_multipliers": {
      "description": "Position size adjustment based on VIX levels",
      "enabled": true,
      "ranges": [
        {
          "vix_min": 0,
          "vix_max": 15,
          "multiplier": 1.0,
          "description": "Low volatility - normal sizing"
        },
        {
          "vix_min": 15,
          "vix_max": 25,
          "multiplier": 0.8,
          "description": "Medium volatility - reduced sizing"
        },
        {
          "vix_min": 25,
          "vix_max": 35,
          "multiplier": 0.6,
          "description": "High volatility - significantly reduced sizing"
        },
        {
          "vix_min": 35,
          "vix_max": 50,
          "multiplier": 0.4,
          "description": "Very high volatility - minimal sizing"
        },
        {
          "vix_min": 50,
          "vix_max": 100,
          "multiplier": 0.2,
          "description": "Extreme volatility - emergency sizing"
        }
      ]
    },
    
    "market_regime_adjustments": {
      "description": "Position sizing adjustments based on market regime",
      "enabled": true,
      "regimes": {
        "trending": {
          "multiplier": 1.0,
          "max_positions": 10,
          "correlation_tolerance": 0.7
        },
        "ranging": {
          "multiplier": 0.8,
          "max_positions": 8,
          "correlation_tolerance": 0.6
        },
        "volatile": {
          "multiplier": 0.6,
          "max_positions": 5,
          "correlation_tolerance": 0.5
        },
        "crisis": {
          "multiplier": 0.3,
          "max_positions": 3,
          "correlation_tolerance": 0.3
        }
      }
    },
    
    "time_based_adjustments": {
      "description": "Time-based position sizing modifications",
      "enabled": true,
      "adjustments": {
        "market_open": {
          "time_range": "09:15:00-09:45:00",
          "multiplier": 0.7,
          "reason": "Higher opening volatility"
        },
        "market_close": {
          "time_range": "15:00:00-15:30:00",
          "multiplier": 0.8,
          "reason": "Closing volatility and liquidity concerns"
        },
        "expiry_day": {
          "time_range": "09:15:00-15:29:30",
          "multiplier": 0.5,
          "reason": "Expiry day increased volatility"
        },
        "news_events": {
          "pause_before_minutes": 15,
          "pause_after_minutes": 15,
          "multiplier": 0.0,
          "reason": "News-driven volatility"
        }
      }
    }
  },
  
  "risk_limits": {
    "description": "Comprehensive risk limit framework",
    
    "account_level_limits": {
      "max_drawdown_percent": 10.0,
      "daily_loss_limit_percent": 5.0,
      "weekly_loss_limit_percent": 15.0,
      "monthly_loss_limit_percent": 25.0,
      "max_portfolio_heat_percent": 20.0,
      "max_leverage": 5.0,
      "margin_utilization_limit": 80.0
    },
    
    "position_level_limits": {
      "max_position_size_percent": 5.0,
      "max_positions_per_symbol": 3,
      "max_total_positions": 15,
      "max_sector_exposure_percent": 25.0,
      "max_market_cap_exposure": {
        "large_cap": 60.0,
        "mid_cap": 30.0,
        "small_cap": 10.0
      }
    },
    
    "time_based_limits": {
      "max_trades_per_day": 50,
      "max_trades_per_hour": 10,
      "max_trades_per_symbol_per_day": 5,
      "cooling_period_minutes": 5,
      "position_hold_time_limits": {
        "min_hold_minutes": 2,
        "max_hold_hours": 8
      }
    },
    
    "correlation_limits": {
      "max_correlation_coefficient": 0.7,
      "max_correlated_positions": 3,
      "correlation_calculation_period": 30,
      "sector_correlation_limit": 0.8,
      "market_correlation_limit": 0.9
    }
  },
  
  "stop_loss_management": {
    "description": "Dynamic stop loss configuration",
    
    "default_settings": {
      "initial_stop_loss_percent": 1.0,
      "max_stop_loss_percent": 3.0,
      "trailing_stop_enabled": true,
      "breakeven_trigger_percent": 0.5,
      "profit_protection_percent": 0.3
    },
    
    "market_regime_stops": {
      "trending_market": {
        "initial_sl_percent": 0.8,
        "trailing_step": 0.2,
        "profit_target_multiplier": 2.0
      },
      "ranging_market": {
        "initial_sl_percent": 1.2,
        "trailing_step": 0.3,
        "profit_target_multiplier": 1.5
      },
      "volatile_market": {
        "initial_sl_percent": 1.5,
        "trailing_step": 0.4,
        "profit_target_multiplier": 1.2
      }
    },
    
    "time_based_stops": {
      "expiry_day_adjustment": {
        "multiplier": 1.5,
        "reason": "Increased gamma risk"
      },
      "eod_stop_adjustment": {
        "time": "15:15:00",
        "multiplier": 1.3,
        "reason": "End of day volatility"
      }
    },
    
    "volatility_adjusted_stops": {
      "enabled": true,
      "vix_based_adjustment": true,
      "atr_based_adjustment": true,
      "min_atr_multiplier": 1.5,
      "max_atr_multiplier": 3.0
    }
  },
  
  "take_profit_management": {
    "description": "Dynamic take profit configuration",
    
    "default_settings": {
      "risk_reward_ratio": 2.0,
      "partial_profit_levels": [0.5, 1.0, 1.5],
      "partial_profit_percentages": [25, 50, 25],
      "scale_out_enabled": true
    },
    
    "market_condition_targets": {
      "high_momentum": {
        "risk_reward_ratio": 3.0,
        "extend_targets": true
      },
      "low_momentum": {
        "risk_reward_ratio": 1.5,
        "quick_exit": true
      },
      "reversal_signals": {
        "immediate_exit": true,
        "ignore_targets": true
      }
    },
    
    "time_decay_management": {
      "options_theta_protection": {
        "enabled": true,
        "theta_threshold": -0.05,
        "time_based_exit": true
      },
      "weekend_exposure": {
        "reduce_positions": true,
        "reduction_percentage": 50.0
      }
    }
  },
  
  "portfolio_risk_management": {
    "description": "Portfolio-level risk controls",
    
    "diversification_requirements": {
      "min_sectors": 3,
      "max_sector_weight": 40.0,
      "min_market_caps": 2,
      "geographical_diversification": false,
      "currency_diversification": false
    },
    
    "concentration_limits": {
      "single_position_limit": 10.0,
      "top_5_positions_limit": 40.0,
      "sector_concentration_limit": 30.0,
      "strategy_concentration_limit": 50.0
    },
    
    "portfolio_rebalancing": {
      "frequency": "daily",
      "rebalance_threshold": 5.0,
      "auto_rebalance": true,
      "rebalance_time": "14:00:00"
    },
    
    "hedging_strategies": {
      "portfolio_hedging": {
        "enabled": false,
        "hedge_ratio": 0.3,
        "hedge_instruments": ["NIFTY_PUT", "VIX_CALL"]
      },
      "pair_trading": {
        "enabled": false,
        "max_pairs": 5,
        "correlation_threshold": 0.8
      }
    }
  },
  
  "drawdown_management": {
    "description": "Drawdown monitoring and response protocols",
    
    "drawdown_thresholds": {
      "warning_level": 3.0,
      "concern_level": 5.0,
      "action_level": 7.5,
      "emergency_level": 10.0
    },
    
    "response_protocols": {
      "warning_level": {
        "actions": ["increased_monitoring", "alert_notifications"],
        "position_size_reduction": 0.0,
        "new_position_restriction": false
      },
      "concern_level": {
        "actions": ["detailed_analysis", "risk_review", "performance_assessment"],
        "position_size_reduction": 20.0,
        "new_position_restriction": false
      },
      "action_level": {
        "actions": ["position_reduction", "strategy_review", "emergency_meeting"],
        "position_size_reduction": 50.0,
        "new_position_restriction": true
      },
      "emergency_level": {
        "actions": ["emergency_stop", "position_closure", "system_shutdown"],
        "position_size_reduction": 100.0,
        "new_position_restriction": true
      }
    },
    
    "recovery_protocols": {
      "gradual_scaling": true,
      "recovery_multipliers": [0.5, 0.7, 0.8, 1.0],
      "performance_gates": [1.0, 2.0, 3.0, 5.0],
      "review_frequency": "daily"
    }
  },
  
  "volatility_risk_management": {
    "description": "Volatility-based risk controls",
    
    "vix_monitoring": {
      "enabled": true,
      "data_source": "NSE_INDIA_VIX",
      "update_frequency": "real_time",
      "historical_period": 252
    },
    
    "volatility_regimes": {
      "low_vol": {
        "vix_range": [0, 15],
        "position_multiplier": 1.0,
        "max_positions": 15
      },
      "normal_vol": {
        "vix_range": [15, 25],
        "position_multiplier": 0.8,
        "max_positions": 12
      },
      "high_vol": {
        "vix_range": [25, 35],
        "position_multiplier": 0.6,
        "max_positions": 8
      },
      "extreme_vol": {
        "vix_range": [35, 100],
        "position_multiplier": 0.3,
        "max_positions": 5
      }
    },
    
    "volatility_adjustments": {
      "position_sizing": true,
      "stop_loss_levels": true,
      "take_profit_levels": true,
      "entry_criteria": true,
      "exit_criteria": true
    }
  },
  
  "liquidity_risk_management": {
    "description": "Liquidity risk controls and monitoring",
    
    "liquidity_requirements": {
      "min_avg_volume": 1000000,
      "min_value_traded": 10000000,
      "max_bid_ask_spread": 0.5,
      "min_market_makers": 3
    },
    
    "position_sizing_adjustments": {
      "high_liquidity": 1.0,
      "medium_liquidity": 0.8,
      "low_liquidity": 0.5,
      "illiquid": 0.0
    },
    
    "exit_protocols": {
      "liquidity_deterioration": {
        "immediate_exit": true,
        "scale_out_strategy": true,
        "max_market_impact": 2.0
      },
      "circuit_breaker_response": {
        "pause_trading": true,
        "reassess_positions": true,
        "reduce_exposure": true
      }
    }
  },
  
  "operational_risk_management": {
    "description": "Operational risk controls and procedures",
    
    "system_risk_controls": {
      "redundancy_systems": true,
      "failover_protocols": true,
      "data_backup_frequency": "real_time",
      "recovery_time_objective": 60,
      "recovery_point_objective": 0
    },
    
    "api_risk_management": {
      "connection_monitoring": true,
      "latency_monitoring": true,
      "error_rate_monitoring": true,
      "rate_limit_management": true,
      "failover_brokers": ["zerodha", "upstox"]
    },
    
    "data_quality_controls": {
      "real_time_validation": true,
      "anomaly_detection": true,
      "cross_reference_checking": true,
      "data_staleness_limits": 5
    }
  },
  
  "regulatory_compliance": {
    "description": "Regulatory risk management and compliance",
    
    "sebi_compliance": {
      "position_reporting": true,
      "risk_disclosure": true,
      "audit_trail": true,
      "margin_requirements": true
    },
    
    "exchange_compliance": {
      "nse_requirements": true,
      "bse_requirements": true,
      "position_limits": true,
      "reporting_obligations": true
    },
    
    "risk_disclosures": {
      "daily_var": true,
      "stress_test_results": true,
      "concentration_reports": true,
      "performance_attribution": true
    }
  },
  
  "emergency_protocols": {
    "description": "Emergency risk management procedures",
    
    "trigger_conditions": {
      "market_crash": {
        "index_decline_percent": 5.0,
        "action": "emergency_stop"
      },
      "flash_crash": {
        "rapid_decline_percent": 2.0,
        "time_window_minutes": 5,
        "action": "immediate_position_closure"
      },
      "system_failure": {
        "api_downtime_minutes": 2,
        "action": "switch_to_backup_systems"
      },
      "risk_limit_breach": {
        "drawdown_threshold": 10.0,
        "action": "emergency_stop_and_review"
      }
    },
    
    "escalation_procedures": {
      "level_1": {
        "triggers": ["warning_threshold_breach"],
        "actions": ["automated_alert", "increased_monitoring"],
        "notification_delay": 0
      },
      "level_2": {
        "triggers": ["concern_threshold_breach", "system_anomaly"],
        "actions": ["management_notification", "position_review"],
        "notification_delay": 5
      },
      "level_3": {
        "triggers": ["action_threshold_breach", "major_system_failure"],
        "actions": ["emergency_meeting", "position_reduction"],
        "notification_delay": 1
      },
      "level_4": {
        "triggers": ["emergency_threshold_breach", "critical_system_failure"],
        "actions": ["complete_shutdown", "regulatory_notification"],
        "notification_delay": 0
      }
    },
    
    "communication_protocols": {
      "internal_alerts": {
        "email": true,
        "sms": true,
        "whatsapp": true,
        "phone_call": true
      },
      "external_notifications": {
        "broker_notification": true,
        "regulatory_reporting": true,
        "client_communication": false
      }
    }
  },
  
  "monitoring_and_reporting": {
    "description": "Risk monitoring and reporting configuration",
    
    "real_time_monitoring": {
      "enabled": true,
      "update_frequency": "tick_by_tick",
      "dashboard_refresh": 1,
      "alert_thresholds": "dynamic"
    },
    
    "reporting_schedule": {
      "intraday_reports": {
        "frequency": "hourly",
        "times": ["10:00", "12:00", "14:00", "15:30"]
      },
      "daily_reports": {
        "time": "16:00:00",
        "include_performance": true,
        "include_risk_metrics": true
      },
      "weekly_reports": {
        "day": "friday",
        "time": "17:00:00",
        "comprehensive_analysis": true
      },
      "monthly_reports": {
        "day": "last_working_day",
        "time": "18:00:00",
        "full_audit": true
      }
    },
    
    "key_risk_metrics": {
      "var_95": true,
      "var_99": true,
      "expected_shortfall": true,
      "maximum_drawdown": true,
      "sharpe_ratio": true,
      "sortino_ratio": true,
      "calmar_ratio": true,
      "correlation_matrix": true,
      "beta_analysis": true,
      "volatility_analysis": true
    }
  },
  
  "backtesting_and_validation": {
    "description": "Risk model validation and backtesting",
    
    "validation_frequency": "weekly",
    "historical_data_period": 252,
    "stress_testing": {
      "enabled": true,
      "scenarios": ["2008_crisis", "2020_covid", "2011_eurozone", "custom_scenarios"],
      "frequency": "monthly"
    },
    
    "model_validation": {
      "var_backtesting": true,
      "correlation_stability": true,
      "volatility_forecasting": true,
      "regime_detection": true
    }
  },
  
  "performance_attribution": {
    "description": "Risk-adjusted performance measurement",
    
    "attribution_factors": {
      "market_exposure": true,
      "sector_allocation": true,
      "security_selection": true,
      "timing_effects": true,
      "volatility_timing": true
    },
    
    "benchmark_comparison": {
      "primary_benchmark": "NIFTY50",
      "secondary_benchmarks": ["NIFTY_MIDCAP", "NIFTY_SMALLCAP"],
      "risk_adjusted_metrics": true
    }
  },
  
  "validation": {
    "last_updated": "2025-01-01T00:00:00Z",
    "validated_by": "EA GlobalFlow Pro Risk Team",
    "validation_status": "ACTIVE",
    "next_review": "2025-01-08T00:00:00Z",
    "approval_status": "APPROVED",
    "risk_framework_version": "v0.1",
    "compliance_verified": true
  }
}