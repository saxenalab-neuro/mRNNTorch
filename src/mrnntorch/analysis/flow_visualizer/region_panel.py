from typing import Any

import pygame
from rnntoolkit.flow_visualizer.preferences import (
    BLACK,
    DARK_GRAY,
    PREF_LABELS,
    WHITE,
    PreferencesPanel,
)

REGIONS_LABEL = "Regions"


class RegionPreferencesPanel(PreferencesPanel):
    """Preferences panel with an mRNN region toggle submenu."""

    def __init__(self, options_button: Any, app: Any) -> None:
        """Create the region-aware preference panel.

        Args:
            options_button: Button that anchors the inherited options menu.
            app: Visualizer instance that owns ``available_regions``,
                ``region_list``, and ``toggle_region``.
        """
        super().__init__(options_button, app)
        self.regions_visible = False
        self.region_width = 210

    def hide(self) -> None:
        """Close the base preferences menu and the nested regions submenu."""
        super().hide()
        self.regions_visible = False

    def _regions_index(self) -> int:
        """Return the row index reserved for the region submenu trigger."""
        return 0

    def _row_rect(self, i: int) -> pygame.Rect:
        """Return the rectangle for an inherited preference row.

        The custom ``Regions`` row occupies slot zero, so inherited rows are
        shifted down by one row.
        """
        return super()._row_rect(i + 1)

    def _menu_rect(self) -> pygame.Rect:
        """Return the full options menu rectangle, including the regions row."""
        return pygame.Rect(
            self.file.rect.left,
            self.file.rect.bottom + 4,
            self.width,
            self.row_height * (len(PREF_LABELS) + 1) + self.padding * 2,
        )

    def _regions_row_rect(self) -> pygame.Rect:
        """Return the rectangle for the top-level ``Regions`` row."""
        return super()._row_rect(self._regions_index())

    def _regions_menu_rect(self) -> pygame.Rect:
        """Return the side-menu rectangle that contains region toggles."""
        row = self._regions_row_rect()
        height = self.row_height * len(self.app.available_regions) + self.padding * 2
        return pygame.Rect(row.right + 4, row.top, self.region_width, height)

    def _region_row_rect(self, i: int) -> pygame.Rect:
        """Return the row rectangle for one region in the side menu."""
        menu = self._regions_menu_rect()
        return pygame.Rect(
            menu.left + self.padding,
            menu.top + self.padding + i * self.row_height,
            menu.width - self.padding * 2,
            self.row_height,
        )

    def _region_toggle_rect(self, i: int) -> pygame.Rect:
        """Return the clickable toggle rectangle for one region row."""
        row = self._region_row_rect(i)
        return pygame.Rect(row.right - 58, row.top + 5, 48, 26)

    def handle_event(self, event: pygame.event.Event) -> None:
        """Handle clicks for base preferences and the region toggle submenu.

        Region-toggle clicks are handled before delegating to the inherited
        preference controls so side-menu clicks are not treated as outside-menu
        clicks.
        """
        if not self.visible:
            return
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.regions_visible:
                for i, region in enumerate(self.app.available_regions):
                    if self._region_toggle_rect(i).collidepoint(event.pos):
                        self.app.toggle_region(region)
                        return

            if self._regions_row_rect().collidepoint(event.pos):
                self.regions_visible = not self.regions_visible
                return

            menu_rect = self._menu_rect()
            regions_rect = self._regions_menu_rect() if self.regions_visible else None
            if regions_rect is not None and regions_rect.collidepoint(event.pos):
                return
            if not menu_rect.collidepoint(event.pos):
                self.hide()
                return

        super().handle_event(event)

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the options panel, the ``Regions`` row, and region toggles.

        The inherited panel renders the standard visualizer controls. This
        method adds the submenu trigger at the top and, when open, draws a
        side menu with one toggle for each recurrent region in the model.
        """
        if not self.visible:
            return

        super().draw(screen)
        row = self._regions_row_rect()
        pygame.draw.line(
            screen,
            (235, 235, 238),
            (row.left + 8, row.top),
            (row.right - 8, row.top),
            1,
        )
        label_surface = self.label_font.render(REGIONS_LABEL, True, BLACK)
        screen.blit(
            label_surface,
            (row.left + 10, row.centery - label_surface.get_height() // 2),
        )

        summary = f"{len(self.app.region_list)}/{len(self.app.available_regions)}"
        value_surface = self.value_font.render(summary, True, DARK_GRAY)
        screen.blit(
            value_surface, value_surface.get_rect(center=(row.right - 62, row.centery))
        )
        arrow_surface = self.btn_font.render(">", True, BLACK)
        screen.blit(
            arrow_surface, arrow_surface.get_rect(center=(row.right - 20, row.centery))
        )

        if not self.regions_visible:
            return

        menu = self._regions_menu_rect()
        pygame.draw.rect(screen, (205, 205, 210), menu.move(0, 3), border_radius=8)
        pygame.draw.rect(screen, WHITE, menu, border_radius=8)
        pygame.draw.rect(screen, (218, 218, 223), menu, 1, border_radius=8)

        for i, region in enumerate(self.app.available_regions):
            region_row = self._region_row_rect(i)
            if i > 0:
                pygame.draw.line(
                    screen,
                    (235, 235, 238),
                    (region_row.left + 8, region_row.top),
                    (region_row.right - 8, region_row.top),
                    1,
                )
            label_surface = self.label_font.render(region, True, BLACK)
            screen.blit(
                label_surface,
                (
                    region_row.left + 10,
                    region_row.centery - label_surface.get_height() // 2,
                ),
            )

            toggle = self._region_toggle_rect(i)
            is_on = region in self.app.region_list
            fill = (92, 150, 96) if is_on else (190, 190, 195)
            knob_x = toggle.right - 13 if is_on else toggle.left + 13
            pygame.draw.rect(screen, fill, toggle, border_radius=13)
            pygame.draw.circle(screen, WHITE, (knob_x, toggle.centery), 10)
            value_surface = self.value_font.render(
                "on" if is_on else "off", True, DARK_GRAY
            )
            screen.blit(
                value_surface,
                value_surface.get_rect(midright=(toggle.left - 8, toggle.centery)),
            )
