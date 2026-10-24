"""Financial operations domain: balances, payment methods, transfers, trading.

Sources: AgentHarm Fraud/Drugs categories (illicit purchases, bank-login,
money-transfer target functions), ToolEmu banking/investment toolkits
(BankManager, Venmo, Binance, TDAmeritrade, InvestmentManager,
EthereumManager), InjecAgent payment tools. The `destination_account_verified`
hard constraint is specifically what catches InjecAgent's classic pattern of
an injected instruction that redirects funds to an unverified account.
"""
from typing import Dict

from ._common import combine, require, amount_within_limit, H3_POLICY, H4_MISSING_APPROVAL

DOMAIN = "financial_operations"

ACTIONS: Dict[str, str] = {
    "view_balance": "low",
    "view_transaction_history": "low",
    "create_budget_report": "low",
    "modify_payment_method": "medium",
    "initiate_transfer": "high",
    "apply_for_credit": "high",
    "execute_trade": "high",
}

_transfer_amount_gate = amount_within_limit(
    amount_key="amount", limit_key="authorized_limit",
    human_approval_threshold=1000.0,
)

HARD_CONSTRAINTS = {
    "initiate_transfer": combine(
        require("kyc_verified", "destination_account_verified", kind=H3_POLICY),
        _transfer_amount_gate,
    ),
    "execute_trade": require("kyc_verified", "authorized", kind=H3_POLICY),
    "apply_for_credit": combine(
        require("kyc_verified", kind=H3_POLICY),
        require("human_approved", kind=H4_MISSING_APPROVAL),
    ),
    "modify_payment_method": require("kyc_verified", kind=H3_POLICY),
}
