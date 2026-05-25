"""
Localization Engine for multi-language, multi-region support.
Handles currency, date/time, business idioms, and cultural adaptations.
"""
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class LocalizationConfig:
    """Regional localization configuration."""
    
    LOCALE_CONFIGS = {
        "en": {
            "currency": "$",
            "currency_code": "USD",
            "date_format": "MM/DD/YYYY",
            "time_format": "12h",
            "timezone": "UTC",
            "greeting_style": "formal_casual",
            "number_separator": ",",
            "decimal_separator": ".",
        },
        "ar": {
            "currency": "ر.س",  # Saudi Riyal
            "currency_code": "SAR",
            "date_format": "DD/MM/YYYY",
            "time_format": "24h",
            "timezone": "Asia/Riyadh",
            "greeting_style": "formal",
            "number_separator": ".",
            "decimal_separator": ",",
        },
        "es": {
            "currency": "€",
            "currency_code": "EUR",
            "date_format": "DD/MM/YYYY",
            "time_format": "24h",
            "timezone": "Europe/Madrid",
            "greeting_style": "formal",
            "number_separator": ".",
            "decimal_separator": ",",
        },
        "fr": {
            "currency": "€",
            "currency_code": "EUR",
            "date_format": "DD/MM/YYYY",
            "time_format": "24h",
            "timezone": "Europe/Paris",
            "greeting_style": "formal",
            "number_separator": " ",
            "decimal_separator": ",",
        },
        "de": {
            "currency": "€",
            "currency_code": "EUR",
            "date_format": "DD.MM.YYYY",
            "time_format": "24h",
            "timezone": "Europe/Berlin",
            "greeting_style": "formal",
            "number_separator": ".",
            "decimal_separator": ",",
        },
        "zh": {
            "currency": "¥",
            "currency_code": "CNY",
            "date_format": "YYYY/MM/DD",
            "time_format": "24h",
            "timezone": "Asia/Shanghai",
            "greeting_style": "formal_respectful",
            "number_separator": ",",
            "decimal_separator": ".",
        },
        "ja": {
            "currency": "¥",
            "currency_code": "JPY",
            "date_format": "YYYY年MM月DD日",
            "time_format": "24h",
            "timezone": "Asia/Tokyo",
            "greeting_style": "formal_respectful",
            "number_separator": ",",
            "decimal_separator": ".",
        },
    }
    
    BUSINESS_IDIOMS = {
        "en": {
            "greeting": "Hello! How can I assist you today?",
            "acknowledge": "I understand your request.",
            "closing": "Thank you for contacting us. Have a great day!",
        },
        "ar": {
            "greeting": "مرحبا! كيف يمكنني مساعدتك اليوم؟",
            "acknowledge": "أفهم طلبك.",
            "closing": "شكراً لتواصلك معنا. وداعاً!",
        },
        "es": {
            "greeting": "¡Hola! ¿Cómo puedo ayudarte hoy?",
            "acknowledge": "Entiendo tu solicitud.",
            "closing": "Gracias por contactarnos. ¡Que tengas un excelente día!",
        },
        "fr": {
            "greeting": "Bonjour! Comment puis-je vous aider?",
            "acknowledge": "Je comprends votre demande.",
            "closing": "Merci de nous avoir contactés. Bonne journée!",
        },
        "de": {
            "greeting": "Hallo! Wie kann ich dir heute helfen?",
            "acknowledge": "Ich verstehe deine Anfrage.",
            "closing": "Danke, dass du uns kontaktiert hast. Einen schönen Tag!",
        },
        "zh": {
            "greeting": "你好！今天有什么我可以帮助你的吗？",
            "acknowledge": "我理解你的请求。",
            "closing": "感谢您与我们联系。祝您有美好的一天！",
        },
        "ja": {
            "greeting": "こんにちは！今日はどのようにお手伝いできますか？",
            "acknowledge": "ご要望を理解いたしました。",
            "closing": "弊社にご連絡ありがとうございました。良い1日をお過ごしください。",
        },
    }


class LocalizationEngine:
    """
    Localizes responses based on detected language and region.
    Handles formatting of currency, dates, and culturally-appropriate idioms.
    """
    
    def __init__(self):
        self.config = LocalizationConfig()
    
    def get_locale_config(self, language: str) -> Dict[str, Any]:
        """
        Get localization configuration for language.
        
        Args:
            language: Language code (e.g., 'en', 'ar')
        
        Returns:
            Locale configuration dictionary
        """
        return self.config.LOCALE_CONFIGS.get(language, self.config.LOCALE_CONFIGS["en"])
    
    def format_currency(self, amount: float, language: str) -> str:
        """
        Format currency amount according to locale.
        
        Args:
            amount: Numeric currency amount
            language: Language code
        
        Returns:
            Formatted currency string (e.g., "$1,234.56" or "1.234,56 €")
        """
        locale_cfg = self.get_locale_config(language)
        
        currency_symbol = locale_cfg["currency"]
        number_sep = locale_cfg["number_separator"]
        decimal_sep = locale_cfg["decimal_separator"]
        
        # Format number with separators
        formatted_num = f"{amount:,.2f}".replace(",", "TEMPSEP").replace(".", decimal_sep).replace("TEMPSEP", number_sep)
        
        # Position currency based on language
        if language in {"ar", "he"}:  # RTL languages: currency at end
            return f"{formatted_num} {currency_symbol}"
        else:  # LTR languages: currency at start
            return f"{currency_symbol}{formatted_num}"
    
    def format_date(self, dt: datetime, language: str) -> str:
        """
        Format datetime according to locale.
        
        Args:
            dt: Datetime object
            language: Language code
        
        Returns:
            Formatted date string
        """
        locale_cfg = self.get_locale_config(language)
        date_format = locale_cfg["date_format"]
        
        # Map custom format to strftime
        format_mapping = {
            "MM/DD/YYYY": "%m/%d/%Y",
            "DD/MM/YYYY": "%d/%m/%Y",
            "DD.MM.YYYY": "%d.%m.%Y",
            "YYYY/MM/DD": "%Y/%m/%d",
            "YYYY年MM月DD日": "%Y年%m月%d日",
        }
        
        strftime_format = format_mapping.get(date_format, "%m/%d/%Y")
        return dt.strftime(strftime_format)
    
    def format_time(self, dt: datetime, language: str) -> str:
        """
        Format time according to locale (12h vs 24h).
        
        Args:
            dt: Datetime object
            language: Language code
        
        Returns:
            Formatted time string
        """
        locale_cfg = self.get_locale_config(language)
        time_format = locale_cfg["time_format"]
        
        if time_format == "12h":
            return dt.strftime("%I:%M %p")  # e.g., "02:30 PM"
        else:  # 24h
            return dt.strftime("%H:%M")  # e.g., "14:30"
    
    def get_business_idiom(self, key: str, language: str) -> str:
        """
        Get culturally-appropriate business idiom.
        
        Args:
            key: Idiom type ('greeting', 'acknowledge', 'closing')
            language: Language code
        
        Returns:
            Localized idiom string
        """
        idioms = self.config.BUSINESS_IDIOMS.get(language, self.config.BUSINESS_IDIOMS["en"])
        return idioms.get(key, self.config.BUSINESS_IDIOMS["en"].get(key, ""))
    
    def inject_locale_context(self, llm_response: str, language: str, context: Dict[str, Any] = None) -> str:
        """
        Inject locale-specific context into LLM response (post-processing).
        
        Args:
            llm_response: Raw LLM response text
            language: Language code
            context: Optional context dict with variables to inject
        
        Returns:
            Localized response
        """
        if not context:
            context = {}
        
        # Replace template variables with localized values
        result = llm_response
        
        # Currency injection
        if "{{CURRENCY}}" in result:
            amount = context.get("amount", 0)
            result = result.replace("{{CURRENCY}}", self.format_currency(amount, language))
        
        # Date injection
        if "{{DATE}}" in result:
            dt = context.get("datetime", datetime.now())
            result = result.replace("{{DATE}}", self.format_date(dt, language))
        
        # Time injection
        if "{{TIME}}" in result:
            dt = context.get("datetime", datetime.now())
            result = result.replace("{{TIME}}", self.format_time(dt, language))
        
        return result


# Global singleton instance
_engine_instance = None

def get_localization_engine() -> LocalizationEngine:
    """Get or create singleton localization engine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = LocalizationEngine()
    return _engine_instance
