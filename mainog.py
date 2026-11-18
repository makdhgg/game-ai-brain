# main.py
import ollama
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from promtog import (
    STAGE1_PROMPT_TEMPLATE,
    SUSPECT_PROMPT_TEMPLATE,
    CONNECTIONS_PROMPT_TEMPLATE,
    VALIDATOR_PROMPT_TEMPLATE,
)


# ============================================================
# БЛОК 1: МОДЕЛИ ДАННЫХ (Dataclasses)
# ============================================================

@dataclass
class WorldLore:
    """Лор мира — вселенная, эпоха, атмосфера. Задаётся пользователем."""
    universe_name: str
    era: str
    atmosphere: str
    tech_level: str
    social_norms: str
    custom_details: str

    def to_prompt_text(self) -> str:
        return (
            f"ВСЕЛЕННАЯ: {self.universe_name}\n"
            f"ЭПОХА: {self.era}\n"
            f"АТМОСФЕРА: {self.atmosphere}\n"
            f"ТЕХНОЛОГИИ: {self.tech_level}\n"
            f"СОЦИАЛЬНЫЕ НОРМЫ: {self.social_norms}\n"
            f"ДОПОЛНИТЕЛЬНЫЕ ДЕТАЛИ: {self.custom_details}"
        )


@dataclass
class Location:
    type: str
    description: str
    time_of_day: str
    weather: str


@dataclass
class Timeline:
    discovery_time: str
    estimated_crime_time: str
    key_time_windows: list[str] = field(default_factory=list)


@dataclass
class VictimInfo:
    description: str
    condition: str


@dataclass
class CrimeCase:
    """Результат первого этапа — описание преступления."""
    crime_type: str
    location: Location
    timeline: Timeline
    crime_summary: str
    victim_info: VictimInfo
    initial_evidence: list[str] = field(default_factory=list)
    atmosphere_tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> 'CrimeCase':
        required = ['crime_type', 'location', 'timeline', 'crime_summary', 'victim_info']
        missing = [f for f in required if f not in data]
        if missing:
            raise ValueError(
                f"CrimeCase: отсутствуют обязательные поля: {missing}\n\n"
                f"Полученные данные:\n{json.dumps(data, ensure_ascii=False, indent=2)}"
            )
        return cls(
            crime_type=data['crime_type'],
            location=Location(**data['location']),
            timeline=Timeline(**data['timeline']),
            crime_summary=data['crime_summary'],
            victim_info=VictimInfo(**data['victim_info']),
            initial_evidence=data.get('initial_evidence', []),
            atmosphere_tags=data.get('atmosphere_tags', [])
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PersonalityProfile:
    core_traits: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    quirks: list[str] = field(default_factory=list)


@dataclass
class Alibi:
    claimed_whereabouts: str
    alibi_strength: str
    alibi_witness: str


@dataclass
class Knowledge:
    knows_for_sure: list[str] = field(default_factory=list)
    suspects_but_unsure: list[str] = field(default_factory=list)


@dataclass
class Secrets:
    hidden_from_police: list[str] = field(default_factory=list)
    would_die_for: list[str] = field(default_factory=list)


@dataclass
class Motive:
    surface_motive: str
    real_motive: str
    emotional_stake: str


@dataclass
class InterrogationNotes:
    how_to_break: str
    likely_lies: list[str] = field(default_factory=list)
    tells: list[str] = field(default_factory=list)


@dataclass
class Suspect:
    """Один подозреваемый."""
    full_name: str
    age: int
    gender: str
    occupation: str
    relationship_to_victim: str
    financial_status: str
    criminal_history: str
    personality_profile: PersonalityProfile
    appearance: str
    speaking_style: str
    alibi: Alibi
    knowledge: Knowledge
    secrets: Secrets
    motive: Motive
    interrogation_notes: InterrogationNotes
    is_guilty: bool
    id: int = 0

    @classmethod
    def from_dict(cls, data: dict, auto_id: int = 1) -> 'Suspect':
        required_fields = [
            'full_name', 'age', 'gender', 'occupation', 'relationship_to_victim',
            'financial_status', 'criminal_history', 'personality_profile',
            'appearance', 'speaking_style', 'alibi', 'knowledge', 'secrets',
            'motive', 'interrogation_notes', 'is_guilty'
        ]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(
                f"Suspect: отсутствуют обязательные поля: {missing}\n\n"
                f"Полученные данные:\n{json.dumps(data, ensure_ascii=False, indent=2)}"
            )
        return cls(
            id=auto_id,
            full_name=data['full_name'],
            age=int(data['age']),
            gender=data['gender'],
            occupation=data['occupation'],
            relationship_to_victim=data['relationship_to_victim'],
            financial_status=data['financial_status'],
            criminal_history=data['criminal_history'],
            personality_profile=PersonalityProfile(**data['personality_profile']),
            appearance=data['appearance'],
            speaking_style=data['speaking_style'],
            alibi=Alibi(**data['alibi']),
            knowledge=Knowledge(**data['knowledge']),
            secrets=Secrets(**data['secrets']),
            motive=Motive(**data['motive']),
            interrogation_notes=InterrogationNotes(**data['interrogation_notes']),
            is_guilty=bool(data['is_guilty'])
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SuspectConnection:
    person_a: int
    person_b: int
    relationship: str
    tension_level: str


@dataclass
class SuspectsGroup:
    """Результат второго этапа — все подозреваемые и связи."""
    suspects: list[Suspect] = field(default_factory=list)
    suspect_connections: list[SuspectConnection] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> 'SuspectsGroup':
        suspects = [Suspect.from_dict(s, auto_id=i + 1) for i, s in enumerate(data['suspects'])]
        connections = [SuspectConnection(**c) for c in data.get('suspect_connections', [])]
        return cls(suspects=suspects, suspect_connections=connections)

    def to_dict(self) -> dict:
        return {
            'suspects': [s.to_dict() for s in self.suspects],
            'suspect_connections': [asdict(c) for c in self.suspect_connections]
        }

    def get_guilty(self) -> Optional[Suspect]:
        """Возвращает настоящего виновного."""
        for s in self.suspects:
            if s.is_guilty:
                return s
        return None


# ============================================================
# БЛОК 2: КЛАСС-ГЕНЕРАТОР
# ============================================================

class CaseGenerator:
    """Главный класс, управляющий многоэтапной генерацией дела."""

    def __init__(
        self,
        world_lore: WorldLore,
        model: str = 'qwen3.5:9b',
        temperature_stage1: float = 0.9,
        temperature_stage2: float = 0.85,
        num_gpu: int = 20,
        num_suspects: int = 4
    ):
        self.world_lore = world_lore
        self.model = model
        self.temperature_stage1 = temperature_stage1
        self.temperature_stage2 = temperature_stage2
        self.num_gpu = num_gpu
        self.num_suspects = num_suspects

        self.crime_case: Optional[CrimeCase] = None
        self.suspects_group: Optional[SuspectsGroup] = None
        self.validation_issues: list[dict] = []

    def _normalize_keys(self, obj):
        """Рекурсивно убирает пробелы по краям ключей JSON."""
        if isinstance(obj, dict):
            return {str(k).strip(): self._normalize_keys(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._normalize_keys(x) for x in obj]
        return obj

    def _call_llm(
        self,
        prompt: str,
        temperature: float,
        max_retries: int = 3,
        expected_keys: list[str] = None,
        num_predict: int = 2048,
        num_ctx: int = 16384
    ) -> dict:
        """Вызов LLM с JSON-форматом, retry, нормализацией ключей и проверкой структуры."""
        last_raw = ""

        for attempt in range(1, max_retries + 1):
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[{'role': 'user', 'content': prompt}],
                    format='json',
                    options={
                        'num_gpu': self.num_gpu,
                        'num_ctx': num_ctx,
                        'num_predict': num_predict,
                        'temperature': temperature,
                    },
                    think=False,
                )

                raw_json = response['message']['content']
                last_raw = raw_json

                data = json.loads(raw_json)
                data = self._normalize_keys(data)

                if expected_keys:
                    missing = [k for k in expected_keys if k not in data]
                    if missing:
                        print(f"   ⚠️ Попытка {attempt}/{max_retries}: отсутствуют ключи {missing}")
                        if attempt == max_retries:
                            raise ValueError(
                                f"Модель не вернула обязательные ключи: {missing}\n\n"
                                f"Последний ответ:\n{last_raw}"
                            )
                        continue

                return data

            except json.JSONDecodeError as e:
                print(f"   ⚠️ Попытка {attempt}/{max_retries}: невалидный JSON: {e}")
                if attempt == max_retries:
                    raise ValueError(
                        f"LLM вернула невалидный JSON после {max_retries} попыток.\n\n"
                        f"Последний ответ:\n{last_raw}"
                    )

            except TypeError:
                # Старая версия ollama может не поддерживать think=False
                try:
                    response = ollama.chat(
                        model=self.model,
                        messages=[{'role': 'user', 'content': prompt}],
                        format='json',
                        options={
                            'num_gpu': self.num_gpu,
                            'num_ctx': num_ctx,
                            'num_predict': num_predict,
                            'temperature': temperature,
                        },
                    )
                    raw_json = response['message']['content']
                    last_raw = raw_json
                    data = json.loads(raw_json)
                    data = self._normalize_keys(data)

                    if expected_keys:
                        missing = [k for k in expected_keys if k not in data]
                        if missing:
                            print(f"   ⚠️ Попытка {attempt}/{max_retries}: отсутствуют ключи {missing}")
                            if attempt == max_retries:
                                raise ValueError(
                                    f"Модель не вернула обязательные ключи: {missing}\n\n"
                                    f"Последний ответ:\n{last_raw}"
                                )
                            continue

                    return data
                except json.JSONDecodeError as e:
                    print(f"   ⚠️ Попытка {attempt}/{max_retries}: невалидный JSON: {e}")
                    if attempt == max_retries:
                        raise ValueError(
                            f"LLM вернула невалидный JSON после {max_retries} попыток.\n\n"
                            f"Последний ответ:\n{last_raw}"
                        )

            except Exception as e:
                print(f"   ⚠️ Попытка {attempt}/{max_retries}: ошибка: {e}")
                if attempt == max_retries:
                    raise

    def generate_crime(self) -> CrimeCase:
        """Этап 1: Генерация базовой ситуации преступления."""
        print(f"ЭТАП 1: Генерация преступления в сеттинге «{self.world_lore.universe_name}»")
        start_time = time.time()

        prompt = STAGE1_PROMPT_TEMPLATE.format(
            world_lore=self.world_lore.to_prompt_text()
        )

        expected_keys = [
            'crime_type', 'location', 'timeline', 'crime_summary',
            'victim_info', 'initial_evidence', 'atmosphere_tags'
        ]
        data = self._call_llm(
            prompt,
            self.temperature_stage1,
            expected_keys=expected_keys,
            num_predict=2048
        )
        self.crime_case = CrimeCase.from_dict(data)

        elapsed = time.time() - start_time
        print(f"Преступление сгенерировано за {elapsed:.1f} сек")
        print(f"   Тип: {self.crime_case.crime_type}")
        print(f"   Место: {self.crime_case.location.type}")

        return self.crime_case

    def generate_single_suspect(
        self,
        crime: CrimeCase,
        suspect_number: int,
        is_guilty: bool,
        previous_suspects: list[Suspect]
    ) -> Suspect:
        """Генерация одного подозреваемого отдельным вызовом LLM с якорями фактов."""
        guilty_status = "ВИНОВНЫЙ" if is_guilty else "НЕВИННЫЙ"

        if previous_suspects:
            previous_text = "\n".join([
                f"- ID {s.id}: {s.full_name}, {s.age}, {s.gender}, "
                f"профессия: {s.occupation}, связь с жертвой: {s.relationship_to_victim}"
                for s in previous_suspects
            ])
        else:
            previous_text = "Пока нет. Это первый подозреваемый."

        # Извлекаем ключевые факты для якорей
        estimated_crime_time = crime.timeline.estimated_crime_time
        discovery_time = crime.timeline.discovery_time
        location_type = crime.location.type
        location_description = crime.location.description
        victim_description = crime.victim_info.description
        victim_condition = crime.victim_info.condition
        weather = crime.location.weather
        tech_level = self.world_lore.tech_level

        # Форматируем список улик
        evidence_list = "\n".join([
            f"   - {ev}" for ev in crime.initial_evidence
        ])

        prompt = SUSPECT_PROMPT_TEMPLATE.format(
            world_lore=self.world_lore.to_prompt_text(),
            estimated_crime_time=estimated_crime_time,
            discovery_time=discovery_time,
            location_type=location_type,
            location_description=location_description,
            victim_description=victim_description,
            victim_condition=victim_condition,
            evidence_list=evidence_list,
            tech_level=tech_level,
            weather=weather,
            crime_case=json.dumps(crime.to_dict(), ensure_ascii=False, indent=2),
            suspect_number=suspect_number,
            total_suspects=self.num_suspects,
            guilty_status=guilty_status,
            previous_suspects=previous_text,
            is_guilty_bool="true" if is_guilty else "false"
        )

        expected_keys = [
            'full_name', 'age', 'gender', 'occupation', 'relationship_to_victim',
            'financial_status', 'criminal_history', 'personality_profile',
            'appearance', 'speaking_style', 'alibi', 'knowledge', 'secrets',
            'motive', 'interrogation_notes', 'is_guilty'
        ]
        data = self._call_llm(
            prompt,
            self.temperature_stage2,
            expected_keys=expected_keys,
            num_predict=3072  # Увеличенный лимит на ответ
        )

        # Защита: если модель всё равно вернула {"suspects": [...]}, достаём первого
        if "suspects" in data and isinstance(data["suspects"], list) and data["suspects"]:
            data = data["suspects"][0]

        # Жёстко фиксируем is_guilty
        data["is_guilty"] = is_guilty

        try:
            return Suspect.from_dict(data, auto_id=suspect_number)
        except (KeyError, TypeError, ValueError) as e:
            print(f"   ❌ Ошибка парсинга подозреваемого #{suspect_number}: {e}")
            print(f"   Сырой JSON от модели:")
            print(json.dumps(data, ensure_ascii=False, indent=2))
            raise

    def generate_connections(self, suspects: list[Suspect]) -> list[SuspectConnection]:
        """Генерация связей между подозреваемыми с якорями установленных данных."""
        # Форматируем установленные факты для якорей
        established_facts = "\n".join([
            f"ID {s.id}: {s.full_name}\n"
            f"   Профессия: {s.occupation}\n"
            f"   Связь с жертвой: {s.relationship_to_victim}\n"
            f"   Видимый мотив: {s.motive.surface_motive}\n"
            f"   Черты характера: {', '.join(s.personality_profile.core_traits)}"
            for s in suspects
        ])

        suspects_brief = [
            {
                "id": s.id,
                "full_name": s.full_name,
                "age": s.age,
                "gender": s.gender,
                "occupation": s.occupation,
                "relationship_to_victim": s.relationship_to_victim,
                "financial_status": s.financial_status,
                "criminal_history": s.criminal_history,
                "surface_motive": s.motive.surface_motive,
                "alibi": s.alibi.claimed_whereabouts,
                "public_traits": s.personality_profile.core_traits,
            }
            for s in suspects
        ]

        prompt = CONNECTIONS_PROMPT_TEMPLATE.format(
            world_lore=self.world_lore.to_prompt_text(),
            established_facts=established_facts,
            suspects_json=json.dumps(suspects_brief, ensure_ascii=False, indent=2)
        )

        data = self._call_llm(
            prompt,
            temperature=0.6,  # Низкая температура для логичных связей
            num_predict=1024
        )

        connections = []
        valid_ids = {s.id for s in suspects}

        for c in data.get("suspect_connections", []):
            try:
                person_a = int(c["person_a"])
                person_b = int(c["person_b"])

                if person_a not in valid_ids or person_b not in valid_ids:
                    continue
                if person_a == person_b:
                    continue

                connections.append(
                    SuspectConnection(
                        person_a=person_a,
                        person_b=person_b,
                        relationship=str(c.get("relationship", "")),
                        tension_level=str(c.get("tension_level", "medium"))
                    )
                )
            except Exception:
                continue

        return connections

    def generate_suspects(self, crime: Optional[CrimeCase] = None) -> SuspectsGroup:
        """Этап 2: Генерация подозреваемых по одному."""
        if crime is None:
            if self.crime_case is None:
                raise ValueError("Сначала нужно сгенерировать преступление (generate_crime)")
            crime = self.crime_case

        print(f"ЭТАП 2: Создание круга подозреваемых ({self.num_suspects} человек)")
        start_time = time.time()

        suspects = []

        for i in range(self.num_suspects):
            suspect_number = i + 1
            is_guilty = (i == 0)  # Первый — виновный

            status = "ВИНОВНЫЙ" if is_guilty else "НЕВИННЫЙ"
            print(f"   Генерация подозреваемого #{suspect_number} ({status})...")

            suspect = self.generate_single_suspect(
                crime=crime,
                suspect_number=suspect_number,
                is_guilty=is_guilty,
                previous_suspects=suspects
            )

            suspects.append(suspect)
            print(f"   ✓ {suspect.full_name} — {suspect.occupation}")

        print("   Генерация связей между подозреваемыми...")
        connections = self.generate_connections(suspects)
        print(f"   ✓ Создано связей: {len(connections)}")

        self.suspects_group = SuspectsGroup(
            suspects=suspects,
            suspect_connections=connections
        )

        elapsed = time.time() - start_time
        print(f"Подозреваемые сгенерированы за {elapsed:.1f} сек")
        print(f"   Количество: {len(self.suspects_group.suspects)}")

        guilty = self.suspects_group.get_guilty()
        if guilty:
            print(f"   Виновный: {guilty.full_name}")

        return self.suspects_group

    def validate_case(
        self,
        crime: CrimeCase,
        suspects_group: SuspectsGroup,
        max_retries: int = 2
    ) -> tuple[SuspectsGroup, list[dict]]:
        """
        Этап 3: Валидация дела на логические несоответствия.
        Валидатор возвращает ТОЛЬКО изменённых подозреваемых, и мы применяем их по id.
        """
        print("ЭТАП 3: Валидация дела на логические несоответствия...")
        start_time = time.time()

        full_case = {
            "crime": crime.to_dict(),
            "suspects": suspects_group.to_dict()
        }

        prompt = VALIDATOR_PROMPT_TEMPLATE.format(
            world_lore=self.world_lore.to_prompt_text(),
            full_case=json.dumps(full_case, ensure_ascii=False, indent=2)
        )

        expected_keys = ['is_consistent', 'issues', 'corrected_suspects', 'corrected_connections']

        # Увеличенные лимиты для валидатора
        data = self._call_llm(
            prompt,
            temperature=0.2,  # Низкая температура для аналитической проверки
            max_retries=max_retries,
            expected_keys=expected_keys,
            num_predict=8192,   # Больше места для ответа
            num_ctx=32768       # Больше контекста для входа+выхода
        )

        issues = data.get('issues', [])
        is_consistent = data.get('is_consistent', True)

        elapsed = time.time() - start_time
        print(f"Валидация завершена за {elapsed:.1f} сек")
        print(f"   Найдено проблем: {len(issues)}")
        print(f"   Дело консистентно: {'Да' if is_consistent else 'Нет'}")

        if issues:
            for issue in issues:
                severity = issue.get('severity', 'unknown')
                location = issue.get('location', 'unknown')
                description = issue.get('description', '')
                print(f"   [{severity.upper()}] {location}: {description}")

        # Применяем исправления, если они есть
        if not is_consistent and data.get('corrected_suspects'):
            print("   Применяю исправления...")
            try:
                # Берём исходный список подозреваемых
                final_suspects = list(suspects_group.suspects)

                # Обновляем только тех, кто пришёл в corrected_suspects
                for corrected_s_data in data['corrected_suspects']:
                    c_id = corrected_s_data.get('id')

                    # Если id не указан, пытаемся найти по имени
                    if c_id is None:
                        c_name = corrected_s_data.get('full_name', '')
                        for s in final_suspects:
                            if s.full_name == c_name:
                                c_id = s.id
                                break

                    if c_id is None:
                        print(f"   ⚠️ Не удалось определить, кого исправлять (нет id и имя не найдено)")
                        continue

                    try:
                        # Создаём исправленного подозреваемого
                        new_suspect = Suspect.from_dict(corrected_s_data, auto_id=int(c_id))

                        # Ищем его в нашем списке и заменяем
                        for idx, s in enumerate(final_suspects):
                            if s.id == new_suspect.id:
                                final_suspects[idx] = new_suspect
                                print(f"   ✓ Исправлен подозреваемый ID {new_suspect.id} ({new_suspect.full_name})")
                                break
                    except Exception as e:
                        print(f"   ⚠️ Не удалось применить исправление для ID {c_id}: {e}")

                # Связи: заменяем, если пришли новые, иначе оставляем старые
                final_connections = suspects_group.suspect_connections
                if data.get('corrected_connections'):
                    valid_ids = {s.id for s in final_suspects}
                    new_connections = []
                    for c in data['corrected_connections']:
                        try:
                            person_a = int(c["person_a"])
                            person_b = int(c["person_b"])
                            if person_a in valid_ids and person_b in valid_ids and person_a != person_b:
                                new_connections.append(
                                    SuspectConnection(
                                        person_a=person_a,
                                        person_b=person_b,
                                        relationship=str(c.get("relationship", "")),
                                        tension_level=str(c.get("tension_level", "medium"))
                                    )
                                )
                        except Exception:
                            continue
                    if new_connections:
                        final_connections = new_connections
                        print(f"   ✓ Связи обновлены ({len(new_connections)} шт.)")

                corrected_group = SuspectsGroup(
                    suspects=final_suspects,
                    suspect_connections=final_connections
                )

                return corrected_group, issues

            except Exception as e:
                print(f"   ⚠️ Критическая ошибка применения исправлений: {e}")
                print(f"   Возвращаю исходные данные без изменений")
                return suspects_group, issues

        return suspects_group, issues

    def generate_full_case(self) -> tuple[CrimeCase, SuspectsGroup]:
        """Полная генерация дела: преступление → подозреваемые → валидация."""
        crime = self.generate_crime()
        suspects = self.generate_suspects(crime)

        # Этап 3: Валидация
        suspects, issues = self.validate_case(crime, suspects)
        self.validation_issues = issues
        self.suspects_group = suspects

        return crime, suspects

    def save_to_file(self, filename: str = 'generated_case.json') -> None:
        """Сохраняет всё сгенерированное дело в JSON-файл."""
        if self.crime_case is None or self.suspects_group is None:
            raise ValueError("Сначала нужно сгенерировать дело (generate_full_case)")

        data = {
            'world_lore': asdict(self.world_lore),
            'crime_case': self.crime_case.to_dict(),
            'suspects_group': self.suspects_group.to_dict(),
            'validation': {
                'issues': self.validation_issues,
                'issues_count': len(self.validation_issues)
            }
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"\nДело сохранено в {filename}")


# ============================================================
# БЛОК 3: ФУНКЦИИ ВЫВОДА
# ============================================================

def print_crime_summary(crime: CrimeCase):
    """Красивый вывод сводки по преступлению."""
    print("\n" + "=" * 70)
    print("ДЕЛО ОТКРЫТО")
    print("=" * 70)
    print(f"Тип преступления: {crime.crime_type}")
    print(f"\nМесто: {crime.location.type}")
    print(f"   {crime.location.description}")
    print(f"   Время: {crime.timeline.discovery_time} | Погода: {crime.location.weather}")
    print(f"\nЧто произошло:")
    print(f"   {crime.crime_summary}")
    print(f"\nЖертва:")
    print(f"   {crime.victim_info.description}")
    print(f"   Состояние: {crime.victim_info.condition}")
    print(f"\nУлики на месте:")
    for i, ev in enumerate(crime.initial_evidence, 1):
        print(f"   {i}. {ev}")
    if crime.atmosphere_tags:
        print(f"\nАтмосфера: {', '.join(crime.atmosphere_tags)}")


def print_suspects_summary(suspects: SuspectsGroup):
    """Красивый вывод сводки по подозреваемым."""
    print("\n" + "=" * 70)
    print("ПОДОЗРЕВАЕМЫЕ")
    print("=" * 70)

    for s in suspects.suspects:
        guilty_mark = "🔴 ВИновный" if s.is_guilty else "⚪ Невинный"
        print(f"\n[{s.id}] {s.full_name}, {s.age} лет, {s.gender} [{guilty_mark}]")
        print(f"   Профессия: {s.occupation}")
        print(f"   Связь с жертвой: {s.relationship_to_victim}")
        print(f"   Внешность: {s.appearance}")
        print(f"   Характер: {', '.join(s.personality_profile.core_traits)}")
        print(f"   Стиль речи: {s.speaking_style}")
        print(f"   Алиби: {s.alibi.claimed_whereabouts} [{s.alibi.alibi_strength}]")
        print(f"   Видимый мотив: {s.motive.surface_motive}")
        print(f"   Настоящий мотив: {s.motive.real_motive}")

    if suspects.suspect_connections:
        print("\n" + "-" * 70)
        print("СВЯЗИ МЕЖДУ ПОДОЗРЕВАЕМЫМИ:")
        print("-" * 70)
        for c in suspects.suspect_connections:
            print(f"   [{c.person_a}] ↔ [{c.person_b}]: {c.relationship} (напряжение: {c.tension_level})")


# ============================================================
# БЛОК 4: ЗАПУСК
# ============================================================

if __name__ == "__main__":
    world = WorldLore(
        universe_name="Нуарный Детройт 1940-х",
        era="1947 год, послевоенная Америка",
        atmosphere="мрачный нуар, дождливые улицы, джаз из баров, сигаретный дым, неоновые вывески",
        tech_level="технологии 1940-х: печатные машинки, телефоны с дисками, автомобили с хромом, фотоаппараты с магнием",
        social_norms="гендерные роли 40-х, расовая сегрегация, влияние мафии, коррупция в полиции, сухие законы в прошлом но бутлегерство в крови",
        custom_details="Город погряз в коррупции. Местная мафия контролирует профсоюзы. Частные детективы — последние честные люди, но и они не без греха."
    )

    generator = CaseGenerator(
        world_lore=world,
        model='qwen3.5:9b',
        temperature_stage1=0.9,
        temperature_stage2=0.85,
        num_gpu=20,
        num_suspects=4
    )

    # Генерируем всё дело (включая валидацию)
    crime, suspects = generator.generate_full_case()

    # Выводим результаты
    print_crime_summary(crime)
    print_suspects_summary(suspects)

    # Сохраняем в файл
    generator.save_to_file('generated_case.json')

    # Доступ к данным для отладки
    print("\n" + "=" * 70)
    print("ДОСТУП К ДАННЫМ (для отладки)")
    print("=" * 70)
    print(f"\ncrime.crime_type = {crime.crime_type}")
    print(f"crime.location.type = {crime.location.type}")
    print(f"len(suspects.suspects) = {len(suspects.suspects)}")

    first_suspect = suspects.suspects[0]
    print(f"\nfirst_suspect.full_name = {first_suspect.full_name}")
    print(f"first_suspect.knowledge.knows_for_sure = {first_suspect.knowledge.knows_for_sure}")

    guilty = suspects.get_guilty()
    print(f"\nguilty.full_name = {guilty.full_name}")
    print(f"guilty.motive.real_motive = {guilty.motive.real_motive}")

    # Результаты валидации
    if generator.validation_issues:
        print(f"\n{'=' * 70}")
        print(f"РЕЗУЛЬТАТЫ ВАЛИДАЦИИ: найдено {len(generator.validation_issues)} проблем")
        print(f"{'=' * 70}")
        for issue in generator.validation_issues:
            print(f"  [{issue.get('severity', '?')}] {issue.get('location', '?')}: {issue.get('description', '')}")
    else:
        print("\n✅ Валидация пройдена: проблем не найдено")