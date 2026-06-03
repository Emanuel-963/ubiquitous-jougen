"""Configuração white-label para distribuições OEM do IonFlow.

Este módulo centraliza a personalização visual e textual da aplicação,
permitindo trocar nome do produto, fornecedor, logotipos, cores e textos de
suporte sem duplicar a base de código. O objetivo é oferecer uma camada leve
de *branding* que possa ser aplicada tanto no arranque da GUI quanto em fluxos
posteriores, como janelas “Sobre” e telas de splash.

Fluxo recomendado
-----------------
1. Criar um arquivo ``white_label.json`` na raiz do projeto ou ao lado deste
   módulo.
2. Carregar a configuração com :meth:`WhiteLabelConfig.load_default`.
3. Aplicar a configuração à instância principal da GUI usando
   :meth:`WhiteLabelConfig.apply_to_gui`.

O método :meth:`apply_to_gui` foi desenhado para ser tolerante: ele tenta usar
métodos e atributos já existentes na aplicação, sem exigir uma API rígida.
Quando um gancho específico não existe, o método apenas regista a intenção e
mantém os metadados no objeto da aplicação para consumo futuro.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from PIL import Image  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - dependência opcional
    Image = None  # type: ignore[assignment]

try:
    import customtkinter as ctk  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - dependência opcional
    ctk = None  # type: ignore[assignment]

_DEFAULT_FILE_NAME = "white_label.json"
_DEFAULT_ABOUT_TEXT = (
    "IonFlow Pipeline é uma plataforma para análise de impedância, DRT e "
    "fluxos de trabalho eletroquímicos."
)


@dataclass
class WhiteLabelConfig:
    """Representa a configuração OEM de *white-label* da aplicação.

    A dataclass foi mantida simples para facilitar persistência em JSON e uso em
    instaladores OEM. Todos os campos são opcionais no arquivo JSON; qualquer
    campo ausente usa o valor padrão definido aqui.
    """

    product_name: str = "IonFlow Pipeline"
    vendor_name: str = "IonFlow"
    logo_path: Optional[str] = None
    primary_color: str = "#1f6aa5"
    secondary_color: str = "#2d2d2d"
    about_text: str = "..."
    support_email: str = ""
    support_url: str = ""
    hide_ionflow_branding: bool = False
    custom_splash_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serializa a configuração para um dicionário compatível com JSON."""
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        """Grava a configuração atual em disco no formato JSON.

        Parameters
        ----------
        path:
            Caminho de destino do arquivo JSON.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        logger.info("White-label salvo em %s", target)

    @classmethod
    def from_json(cls, path: str | Path) -> "WhiteLabelConfig":
        """Carrega uma configuração *white-label* a partir de JSON.

        Chaves desconhecidas são ignoradas para permitir compatibilidade futura.
        """
        source = Path(path)
        with source.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        known_fields = cls.__dataclass_fields__
        filtered: dict[str, Any] = {}
        for key, value in raw.items():
            if key not in known_fields:
                logger.warning("White-label: chave desconhecida ignorada: %s", key)
                continue
            filtered[key] = value

        config = cls(**filtered)
        logger.info("White-label carregado de %s", source)
        return config

    @classmethod
    def load_default(cls, start_dir: str | Path | None = None) -> "WhiteLabelConfig":
        """Carrega ``white_label.json`` se existir, senão retorna os padrões.

        A pesquisa considera primeiro o diretório informado em ``start_dir``.
        Se nada for encontrado, o método tenta o diretório de trabalho atual e,
        por fim, o diretório deste próprio módulo.
        """
        candidates: list[Path] = []
        if start_dir is not None:
            candidates.append(Path(start_dir) / _DEFAULT_FILE_NAME)

        candidates.extend(
            [
                Path.cwd() / _DEFAULT_FILE_NAME,
                Path(__file__).resolve().parent.parent / _DEFAULT_FILE_NAME,
                Path(__file__).resolve().parent / _DEFAULT_FILE_NAME,
            ]
        )

        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if candidate.exists():
                try:
                    return cls.from_json(candidate)
                except Exception as exc:  # pragma: no cover - caminho excepcional
                    logger.warning(
                        "Falha ao carregar white-label de %s: %s; usando padrão",
                        candidate,
                        exc,
                    )
                    break

        logger.info("Nenhum %s encontrado; usando configuração padrão", _DEFAULT_FILE_NAME)
        return cls()

    def apply_to_gui(self, app: Any) -> None:
        """Aplica a configuração OEM a uma instância de GUI.

        O método tenta atualizar:

        * título da janela principal;
        * metadados e texto do diálogo “Sobre”;
        * recursos visuais da splash screen;
        * atributos auxiliares que a aplicação possa consumir depois.

        A implementação evita falhas duras quando a GUI expõe apenas parte
        desses elementos.
        """
        title = self.product_name.strip() or "IonFlow Pipeline"
        about_message = self._resolved_about_text()
        logo = self._resolve_existing_path(self.logo_path)
        splash = self._resolve_existing_path(self.custom_splash_path) or logo

        self._apply_window_title(app, title)
        self._apply_about_dialog(app, about_message)
        self._apply_splash(app, splash, title)
        self._apply_theme_metadata(app)

        try:
            setattr(app, "white_label_config", self)
        except Exception:
            logger.debug("Não foi possível anexar white_label_config ao app")

    def _resolved_about_text(self) -> str:
        """Monta o texto final do diálogo “Sobre”."""
        base_text = self.about_text.strip() if self.about_text.strip() != "..." else _DEFAULT_ABOUT_TEXT
        branding = "" if self.hide_ionflow_branding else "\nTecnologia base: IonFlow Pipeline."
        support_parts = []
        if self.support_email:
            support_parts.append(f"E-mail de suporte: {self.support_email}")
        if self.support_url:
            support_parts.append(f"Suporte: {self.support_url}")
        support_text = f"\n{' | '.join(support_parts)}" if support_parts else ""
        return (
            f"{self.product_name}\n"
            f"Fornecedor: {self.vendor_name}\n\n"
            f"{base_text}{branding}{support_text}"
        )

    def _apply_window_title(self, app: Any, title: str) -> None:
        """Tenta atualizar o título da janela principal."""
        title_targets = [app, getattr(app, "root", None), getattr(app, "master", None)]
        for target in title_targets:
            if target is None:
                continue
            title_method = getattr(target, "title", None)
            if callable(title_method):
                try:
                    title_method(title)
                    logger.debug("Título da janela atualizado para %s", title)
                    break
                except Exception as exc:
                    logger.debug("Falha ao atualizar título da janela: %s", exc)

        for attr in ("app_title", "window_title", "product_name"):
            try:
                setattr(app, attr, title)
            except Exception:
                logger.debug("Não foi possível definir %s no app", attr)

    def _apply_about_dialog(self, app: Any, about_message: str) -> None:
        """Armazena ou encaminha conteúdo do diálogo “Sobre”."""
        values = {
            "about_text": about_message,
            "about_message": about_message,
            "about_dialog_text": about_message,
            "support_email": self.support_email,
            "support_url": self.support_url,
            "vendor_name": self.vendor_name,
        }
        for attr, value in values.items():
            try:
                setattr(app, attr, value)
            except Exception:
                logger.debug("Não foi possível definir atributo de about: %s", attr)

        for method_name in ("set_about_text", "configure_about_dialog", "set_about_dialog"):
            method = getattr(app, method_name, None)
            if callable(method):
                try:
                    method(about_message)
                    logger.debug("Conteúdo do diálogo 'Sobre' aplicado via %s", method_name)
                    return
                except Exception as exc:
                    logger.debug("Falha ao aplicar about via %s: %s", method_name, exc)

    def _apply_splash(self, app: Any, splash_path: Optional[str], title: str) -> None:
        """Atualiza texto e imagem associados à splash screen, se existirem."""
        if splash_path:
            for attr in ("splash_path", "custom_splash_path", "splash_image_path"):
                try:
                    setattr(app, attr, splash_path)
                except Exception:
                    logger.debug("Não foi possível definir %s no app", attr)

        for attr in ("splash_title", "splash_text"):
            try:
                setattr(app, attr, title)
            except Exception:
                logger.debug("Não foi possível definir %s no app", attr)

        splash_widget = getattr(app, "splash_label", None)
        if splash_widget is not None:
            try:
                splash_widget.configure(text=title)
            except Exception:
                logger.debug("Falha ao atualizar texto da splash_label")
            if splash_path:
                image_obj = self._build_ctk_image(splash_path)
                if image_obj is not None:
                    try:
                        splash_widget.configure(image=image_obj)
                        setattr(app, "_white_label_splash_image", image_obj)
                    except Exception as exc:
                        logger.debug("Falha ao aplicar imagem na splash_label: %s", exc)

        for method_name in ("set_splash", "set_splash_screen", "configure_splash"):
            method = getattr(app, method_name, None)
            if callable(method):
                try:
                    if splash_path:
                        method(splash_path, title)
                    else:
                        method(title)
                    logger.debug("Splash aplicada via %s", method_name)
                    return
                except TypeError:
                    try:
                        method(splash_path)
                        return
                    except Exception as exc:
                        logger.debug("Falha ao aplicar splash via %s: %s", method_name, exc)
                except Exception as exc:
                    logger.debug("Falha ao aplicar splash via %s: %s", method_name, exc)

    def _apply_theme_metadata(self, app: Any) -> None:
        """Expõe metadados de tema para consumo pela GUI."""
        metadata = {
            "primary_color": self.primary_color,
            "secondary_color": self.secondary_color,
            "logo_path": self._resolve_existing_path(self.logo_path),
            "vendor_name": self.vendor_name,
            "hide_ionflow_branding": self.hide_ionflow_branding,
        }
        try:
            setattr(app, "white_label_theme", metadata)
        except Exception:
            logger.debug("Não foi possível definir white_label_theme no app")

        method = getattr(app, "configure_theme", None)
        if callable(method):
            try:
                method(**metadata)
            except Exception as exc:
                logger.debug("configure_theme ignorou metadados white-label: %s", exc)

    @staticmethod
    def _resolve_existing_path(raw_path: Optional[str]) -> Optional[str]:
        """Resolve um caminho existente retornando ``str`` ou ``None``."""
        if not raw_path:
            return None
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if candidate.exists():
            return str(candidate)
        logger.warning("Recurso white-label não encontrado: %s", raw_path)
        return None

    @staticmethod
    def _build_ctk_image(path: str) -> Any | None:
        """Cria uma ``CTkImage`` quando Pillow e CustomTkinter estão disponíveis."""
        if Image is None or ctk is None:
            logger.debug("Pillow/CustomTkinter indisponíveis; imagem de splash ignorada")
            return None
        try:
            image = Image.open(path)
            return ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
        except Exception as exc:
            logger.warning("Falha ao carregar imagem white-label %s: %s", path, exc)
            return None


DEFAULT_WHITE_LABEL_CONFIG = WhiteLabelConfig.load_default()
# Instância carregada automaticamente a partir de ``white_label.json`` quando o
# módulo é importado. Se o arquivo não existir, esta constante contém os valores
# padrão da :class:`WhiteLabelConfig`.
