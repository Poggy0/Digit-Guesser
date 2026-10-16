import threading
import traceback
import tkinter as tk
from tkinter import ttk


BACKGROUND = "#F3F5F9"
FOREGROUND = "#172033"
MUTED = "#667085"
ACCENT = "#4F46E5"
CHART_LINE = "#0F9D8A"


# Create the application window and define reusable widget styles.
def init_interface():
    root = tk.Tk()
    root.title("MNIST Digit Guesser")
    root.geometry("900x760")
    root.minsize(720, 640)
    root.configure(bg=BACKGROUND)

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("App.TFrame", background=BACKGROUND)
    style.configure("Card.TFrame", background="white")
    style.configure("App.TLabel", background=BACKGROUND, foreground=FOREGROUND)
    style.configure("Muted.TLabel", background=BACKGROUND, foreground=MUTED)
    style.configure("Card.TLabel", background="white", foreground=FOREGROUND)
    style.configure("Metric.TLabel", background="white", foreground=ACCENT, font=("TkDefaultFont", 18, "bold"))
    style.configure("Title.TLabel", background=BACKGROUND, foreground=FOREGROUND, font=("TkDefaultFont", 22, "bold"))
    style.configure("Accent.TButton", background=ACCENT, foreground="white", padding=(14, 9))
    style.map("Accent.TButton", background=[("active", "#4338CA"), ("disabled", "#A5B4FC")])
    style.configure("Cancel.TButton", background="#DC2626", foreground="white", padding=(14, 9))
    style.map("Cancel.TButton", background=[("active", "#B91C1C"), ("disabled", "#FCA5A5")])

    return root


# Build the training controls, metric cards, and accuracy chart.
def init_buttons(root, model):
    # Use each pool's actual size to set its initial selection and upper limit.
    training_total = len(model.X_train)
    testing_total = len(model.X_test)

    # Set up the main window layout and training configuration controls.
    container = ttk.Frame(root, style="App.TFrame", padding=28)
    container.pack(fill="both", expand=True)
    container.columnconfigure(0, weight=1)
    container.rowconfigure(4, weight=1)

    ttk.Label(container, text="MNIST Digit Guesser", style="Title.TLabel").grid(
        row=0, column=0, sticky="w"
    )
    ttk.Label(
        container,
        text="Train a small neural network and watch its test accuracy improve.",
        style="Muted.TLabel",
    ).grid(row=1, column=0, sticky="w", pady=(4, 18))

    settings = ttk.LabelFrame(container, text="Training setup", padding=16)
    settings.grid(row=2, column=0, sticky="ew")
    for column in range(4):
        settings.columnconfigure(column, weight=1)

    epochs_var = tk.StringVar(value="5")
    training_var = tk.StringVar(value=str(training_total))
    testing_var = tk.StringVar(value=str(testing_total))
    batch_size_var = tk.StringVar(value="32")
    dataset_summary = tk.StringVar()

    # Add one labeled numeric input to the settings row.
    def add_spinbox(parent, label, variable, maximum, column):
        ttk.Label(parent, text=label, style="App.TLabel").grid(
            row=0, column=column, sticky="w", padx=(0, 12)
        )
        spinbox = ttk.Spinbox(
            parent, from_=1, to=maximum, textvariable=variable, width=12
        )
        spinbox.grid(row=1, column=column, sticky="ew", padx=(0, 12), pady=(6, 0))
        return spinbox

    epochs_input = add_spinbox(settings, "Epochs", epochs_var, 1000, 0)
    training_input = add_spinbox(settings, "Training images", training_var, training_total, 1)
    testing_input = add_spinbox(settings, "Testing images", testing_var, testing_total, 2)
    batch_size_input = add_spinbox(settings, "Batch size", batch_size_var, training_total, 3)

    # Keep the available and selected dataset counts visible as inputs change.
    def refresh_dataset_summary(*_):
        try:
            training_count = int(training_var.get())
            testing_count = int(testing_var.get())
            dataset_summary.set(
                f"Available: {training_total:,} training / {testing_total:,} testing images"
                f"    Selected: {training_count:,} / {testing_count:,}"
            )
        except ValueError:
            dataset_summary.set(
                f"Available: {training_total:,} training / {testing_total:,} testing images"
            )

    for variable in (training_var, testing_var):
        variable.trace_add("write", refresh_dataset_summary)
    refresh_dataset_summary()
    ttk.Label(settings, textvariable=dataset_summary, style="Muted.TLabel").grid(
        row=2, column=0, columnspan=3, sticky="w", pady=(12, 0)
    )

    # Add the action buttons and status text below the setup controls.
    actions = ttk.Frame(container, style="App.TFrame")
    actions.grid(row=3, column=0, sticky="ew", pady=16)
    actions.columnconfigure(0, weight=1)

    train_button = ttk.Button(actions, text="Start training", style="Accent.TButton")
    test_button = ttk.Button(actions, text="Test model")
    train_button.grid(row=0, column=0, sticky="w")
    test_button.grid(row=0, column=1, sticky="w", padx=(10, 0))
    status_var = tk.StringVar(value="Ready to train")
    ttk.Label(actions, textvariable=status_var, style="Muted.TLabel").grid(
        row=0, column=2, sticky="e", padx=(16, 0)
    )

    metrics = ttk.Frame(container, style="App.TFrame")
    metrics.grid(row=4, column=0, sticky="nsew")
    metrics.columnconfigure(0, weight=1)
    metrics.rowconfigure(1, weight=1)

    metric_cards = ttk.Frame(metrics, style="App.TFrame")
    metric_cards.grid(row=0, column=0, sticky="ew", pady=(0, 14))
    metric_cards.columnconfigure(0, weight=1)
    metric_cards.columnconfigure(1, weight=1)
    accuracy_var = tk.StringVar(value="—")
    loss_var = tk.StringVar(value="—")

    # Show the most recent accuracy and loss in summary cards.
    for column, title, value in (
        (0, "Latest test accuracy", accuracy_var),
        (1, "Latest training loss", loss_var),
    ):
        card = ttk.Frame(metric_cards, style="Card.TFrame", padding=14)
        card.grid(row=0, column=column, sticky="ew", padx=(0, 10) if column == 0 else (0, 0))
        ttk.Label(card, text=title, style="Card.TLabel").pack(anchor="w")
        ttk.Label(card, textvariable=value, style="Metric.TLabel").pack(anchor="w", pady=(4, 0))

    chart_card = ttk.Frame(metrics, style="Card.TFrame", padding=14)
    chart_card.grid(row=1, column=0, sticky="nsew")
    chart_card.columnconfigure(0, weight=1)
    chart_card.rowconfigure(1, weight=1)
    ttk.Label(chart_card, text="Test accuracy by epoch", style="Card.TLabel").grid(
        row=0, column=0, sticky="w", pady=(0, 8)
    )
    chart = tk.Canvas(chart_card, background="white", highlightthickness=0, height=280)
    chart.grid(row=1, column=0, sticky="nsew")
    accuracy_history = []
    chart_epoch_count = 5

    # Redraw the accuracy chart using the current canvas size and history.
    def draw_chart(_event=None):
        chart.delete("all")
        width = _event.width if _event is not None else chart.winfo_width()
        height = _event.height if _event is not None else chart.winfo_height()
        if width < 100 or height < 100:
            return

        left, right = 58, width - 18
        top, bottom = 18, height - 34
        for percentage in (0, 25, 50, 75, 100):
            y = bottom - (percentage / 100) * (bottom - top)
            chart.create_line(left, y, right, y, fill="#E7EAF0")
            chart.create_text(
                left - 10, y, text=f"{percentage}%", fill=MUTED, anchor="e",
                font=("TkDefaultFont", 9),
            )

        chart.create_text(
            (left + right) / 2, height - 10, text="Epoch", fill=MUTED,
            font=("TkDefaultFont", 9),
        )
        if not accuracy_history:
            chart.create_text(
                (left + right) / 2, (top + bottom) / 2,
                text="Start training to see test accuracy here",
                fill=MUTED, font=("TkDefaultFont", 10),
            )
            return

        points = []
        for epoch, accuracy in accuracy_history:
            x = left + ((epoch - 1) / max(chart_epoch_count - 1, 1)) * (right - left)
            y = bottom - accuracy * (bottom - top)
            points.append((x, y))
        if len(points) > 1:
            chart.create_line(*[coordinate for point in points for coordinate in point],
                              fill=CHART_LINE, width=3, smooth=True)
        for x, y in points:
            chart.create_oval(x - 4, y - 4, x + 4, y + 4, fill=CHART_LINE, outline="white", width=2)

    chart.bind("<Configure>", draw_chart)
    draw_chart()

    controls = (epochs_input, training_input, testing_input, batch_size_input, test_button)
    training_cancel_event = None

    # Lock settings during work and show the appropriate training action.
    def set_busy(busy, training=False):
        state = "disabled" if busy else "normal"
        for control in controls:
            control.configure(state=state)
        if training and busy:
            train_button.configure(
                text="Cancel training",
                style="Cancel.TButton",
                state="normal",
            )
        else:
            train_button.configure(
                text="Start training",
                style="Accent.TButton",
                state=state,
            )

    # Parse image counts and ensure each stays within its own MNIST pool.
    def selected_image_counts():
        try:
            training_count = int(training_var.get())
            testing_count = int(testing_var.get())
        except ValueError as error:
            raise ValueError("Enter whole numbers for the image counts.") from error

        if not 1 <= training_count <= training_total:
            raise ValueError(f"Training images must be between 1 and {training_total:,}.")
        if not 1 <= testing_count <= testing_total:
            raise ValueError(f"Testing images must be between 1 and {testing_total:,}.")
        return training_count, testing_count

    # Parse and validate the epoch count along with the image counts.
    def selected_counts():
        try:
            epochs = int(epochs_var.get())
            batch_size = int(batch_size_var.get())
        except ValueError as error:
            raise ValueError("Enter whole numbers for epochs and batch size.") from error
        if not 1 <= epochs <= 1000:
            raise ValueError("Epochs must be between 1 and 1,000.")
        if not 1 <= batch_size <= training_total:
            raise ValueError(f"Batch size must be between 1 and {training_total:,}.")
        training_count, testing_count = selected_image_counts()
        return epochs, training_count, testing_count, batch_size

    # Run long operations away from the UI thread and report any failures.
    def start_worker(work, action_name, training=False):
        set_busy(True, training=training)

        def run():
            try:
                work()
            except Exception as error:
                traceback.print_exc()
                root.after(
                    0,
                    lambda message=str(error): status_var.set(f"{action_name} failed: {message}"),
                )
            finally:
                def finish():
                    nonlocal training_cancel_event
                    if training:
                        training_cancel_event = None
                    set_busy(False)

                root.after(0, finish)

        threading.Thread(target=run, daemon=True).start()

    # Validate the settings, reset chart state, and start model training.
    def run_training():
        nonlocal chart_epoch_count, training_cancel_event

        try:
            epochs, training_count, testing_count, batch_size = selected_counts()
        except ValueError as error:
            status_var.set(str(error))
            return

        chart_epoch_count = epochs
        accuracy_history.clear()
        accuracy_var.set("—")
        loss_var.set("—")
        training_cancel_event = threading.Event()
        draw_chart()
        status_var.set("Training in progress...")

        def on_status_update(message):
            root.after(0, lambda: status_var.set(message))

        # Schedule metric changes on Tk's UI thread after each epoch.
        def on_metrics_update(epoch, loss, accuracy, elapsed):
            def update_ui():
                accuracy_history.append((epoch, accuracy))
                accuracy_var.set(f"{accuracy:.1%}")
                loss_var.set(f"{loss:.4f}")
                status_var.set(
                    f"Epoch {epoch}/{epochs} · {elapsed:.1f}s"
                )
                draw_chart()

            root.after(0, update_ui)

        start_worker(
            lambda: model.train(
                epochs,
                on_status_update,
                training_size=training_count,
                testing_size=testing_count,
                metrics_update=on_metrics_update,
                batch_size=batch_size,
                cancel_event=training_cancel_event,
            ),
            "Training",
            training=True,
        )

    # Signal the current batch to finish, then stop before the next batch.
    def handle_training_button():
        nonlocal training_cancel_event
        if training_cancel_event is None:
            run_training()
            return
        training_cancel_event.set()
        train_button.configure(text="Cancelling...", state="disabled")
        status_var.set("Cancelling training after the current batch...")

    # Evaluate the model on the selected number of held-out test images.
    def run_test():
        try:
            _, testing_count = selected_image_counts()
        except ValueError as error:
            status_var.set(str(error))
            return
        status_var.set("Testing in progress...")

        def work():
            accuracy = model.test(testing_count)
            root.after(0, lambda: accuracy_var.set(f"{accuracy:.1%}"))
            root.after(0, lambda: status_var.set(f"Test accuracy: {accuracy:.2%}"))

        start_worker(work, "Testing")

    train_button.configure(command=handle_training_button)
    test_button.configure(command=run_test)
    return container
