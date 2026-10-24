"""Rule-based classifier mapping a tool/function name (and optionally its
toolkit) to a (domain, action, criticality) triple.

Used by the AgentHarm, InjecAgent, ToolEmu and AgentDojo normalizers:
  1. Domain: toolkit lookup table, with a keyword fallback on the function
     name for sources that have no toolkit grouping.
  2. Criticality: verb keywords on the function name (delete/grant/transfer
     -> high, send/post/write -> medium, get/search/query -> low).

tau-bench and SWE-bench do not go through this module.
"""
import re
from typing import Dict, Optional, Tuple

# --- Toolkit -> domain (the toolkit pool shared by InjecAgent and ToolEmu) ---
TOOLKIT_TO_DOMAIN: Dict[str, str] = {
    # software_engineering
    "Terminal": "software_engineering",
    "GitHub": "software_engineering",
    # financial_operations
    "BankManager": "financial_operations",
    "Venmo": "financial_operations",
    "TDAmeritrade": "financial_operations",
    "InvestmentManager": "financial_operations",
    "EthereumManager": "financial_operations",
    "Binance": "financial_operations",
    # customer_service (order/shipping/e-commerce toolkits)
    "Amazon": "customer_service",
    "Expedia": "customer_service",
    "Shopify": "customer_service",
    "FedExShipManager": "customer_service",
    "InventoryManagementSystem": "customer_service",
    # communications_and_social_engineering
    "Gmail": "communications_and_social_engineering",
    "Slack": "communications_and_social_engineering",
    "TwitterManager": "communications_and_social_engineering",
    "FacebookManager": "communications_and_social_engineering",
    "Twilio": "communications_and_social_engineering",
    "DeepfakeGenerator": "communications_and_social_engineering",
    # data_and_systems_access (default-ish bucket: files, IoT/physical
    # access, medical/genetic/identity data, search/read tools)
    "GoogleCalendar": "data_and_systems_access",
    "Todoist": "data_and_systems_access",
    "EvernoteManager": "data_and_systems_access",
    "Dropbox": "data_and_systems_access",
    "GoogleHome": "data_and_systems_access",
    "AugustSmartLock": "data_and_systems_access",
    "IndoorRobot": "data_and_systems_access",
    "TrafficControl": "data_and_systems_access",
    "EmergencyDispatchSystem": "data_and_systems_access",
    "Teladoc": "data_and_systems_access",
    "EpicFHIR": "data_and_systems_access",
    "The23andMe": "data_and_systems_access",
    "Spokeo": "data_and_systems_access",
    "NortonIdentitySafe": "data_and_systems_access",
    "CiscoUmbrella": "data_and_systems_access",
    "IFTTT": "data_and_systems_access",
    "GoogleSearch": "data_and_systems_access",
    "GoogleMap": "data_and_systems_access",
    "WebBrowser": "data_and_systems_access",
}

DEFAULT_DOMAIN = "data_and_systems_access"

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _normalize(name: str) -> str:
    """CamelCase (InjecAgent/ToolEmu tool names) and snake_case (AgentHarm
    target_functions) both normalize to lowercase-with-underscores, so the
    same keyword lists match both naming conventions."""
    return _CAMEL_BOUNDARY.sub("_", name).lower()

# --- Function-name keyword -> domain (fallback when no toolkit is known,
# e.g. AgentHarm's target_functions), checked in this priority order -------
_DOMAIN_KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("financial_operations", (
        "transfer_money", "paypal", "venmo", "bank", "credit_card", "crypto",
        "bitcoin", "ethereum", "binance", "trade", "invest", "wire",
        "submit_alpha_bay_order", "submit_amazon_order", "make_online_purchase",
        "purchase_credit_card", "login_into_bank",
    )),
    ("software_engineering", (
        "terminal", "execute", "command", "script", "deploy", "install",
        "package", "git", "pull_request", "compile", "ransomware", "malware",
        "exploit", "backdoor", "ssh", "download_software",
    )),
    ("communications_and_social_engineering", (
        "post_on", "post_to", "send_email", "send_message", "send_direct_message",
        "dm_", "reply", "comment", "contact", "tweet", "twitter", "facebook",
        "instagram", "tiktok", "youtube", "pinterest", "reddit", "blog",
        "create_ai_video", "stable_diffusion", "whatsapp", "tinder",
        "follow_", "repost_", "report_",
    )),
)

DEFAULT_KEYWORD_DOMAIN = "data_and_systems_access"

# --- Function-name keyword -> criticality, checked high -> medium -> low --
_HIGH_KEYWORDS = (
    "delete", "remove", "purge", "grant", "revoke", "transfer", "wire",
    "purchase", "buy", "execute", "deploy", "install",
    "ransomware", "malware", "exploit", "backdoor", "hack", "export",
    "extract", "login", "credential", "terminal", "command",
    "alpha_bay", "post_on", "post_to", "upload_youtube", "upload_tiktok",
    "unlock", "pay", "dispatch", "override", "disable", "bypass",
    "reset_password", "change_password", "withdraw", "sell", "book",
)
_MEDIUM_KEYWORDS = (
    "send", "post", "create", "write", "edit", "update", "modify", "upload",
    "share", "publish", "message", "follow", "connect", "convert", "lock",
)
_LOW_KEYWORDS = (
    "get", "search", "query", "read", "list", "view", "check", "download",
    "fetch", "find",
)

DEFAULT_CRITICALITY = "medium"  # unrecognized actions default to medium, not low

# Toolkits where any non-read-only action is treated as high criticality
_HIGH_STAKES_TOOLKITS = frozenset({
    "BankManager", "Venmo", "TDAmeritrade", "InvestmentManager",
    "EthereumManager", "Binance", "AugustSmartLock", "GoogleHome",
    "EmergencyDispatchSystem", "TrafficControl", "EpicFHIR", "Teladoc",
    "The23andMe", "NortonIdentitySafe", "DeepfakeGenerator", "CiscoUmbrella",
})


def classify_criticality(function_name: str, toolkit: Optional[str] = None) -> str:
    name = _normalize(function_name)
    for kw in _HIGH_KEYWORDS:
        if kw in name:
            return "high"
    is_low = any(kw in name for kw in _LOW_KEYWORDS)
    # In a high-stakes toolkit even "share" or "send" is high risk
    if toolkit in _HIGH_STAKES_TOOLKITS and not is_low:
        return "high"
    for kw in _MEDIUM_KEYWORDS:
        if kw in name:
            return "medium"
    if is_low:
        return "low"
    return DEFAULT_CRITICALITY


def classify_domain(function_name: str, toolkit: Optional[str] = None) -> str:
    if toolkit and toolkit in TOOLKIT_TO_DOMAIN:
        return TOOLKIT_TO_DOMAIN[toolkit]
    name = _normalize(function_name)
    for domain, keywords in _DOMAIN_KEYWORDS:
        if any(kw in name for kw in keywords):
            return domain
    return DEFAULT_KEYWORD_DOMAIN


# --- Domain + criticality -> representative action name ---
# Hard constraints are keyed by each domain's fixed actions, so every tool is
# mapped onto one action per (domain, criticality) cell.
_TIER_ACTION: Dict[str, Dict[str, str]] = {
    "customer_service": {"low": "view_order", "medium": "modify_order", "high": "execute_refund"},
    "software_engineering": {"low": "read_file", "medium": "write_file", "high": "execute_arbitrary_command"},
    "financial_operations": {"low": "view_transaction_history", "medium": "modify_payment_method", "high": "initiate_transfer"},
    "communications_and_social_engineering": {"low": "draft_message", "medium": "send_external_email", "high": "post_public_content"},
    "data_and_systems_access": {"low": "read_public_record", "medium": "update_record", "high": "export_user_data"},
}


def action_for(domain: str, criticality: str) -> str:
    """Representative action name for a (domain, criticality) cell -- used
    by sources like ToolEmu that describe risk in narrative text rather
    than naming a specific tool/function to classify."""
    return _TIER_ACTION[domain][criticality]


def classify(function_name: str, toolkit: Optional[str] = None) -> Tuple[str, str, str]:
    """Returns (domain, action, criticality) for a raw tool/function name."""
    domain = classify_domain(function_name, toolkit)
    criticality = classify_criticality(function_name, toolkit)
    action = _TIER_ACTION[domain][criticality]
    return domain, action, criticality
