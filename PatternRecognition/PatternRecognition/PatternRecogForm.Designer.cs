namespace PatternRecognition
{
    partial class PatternRecognition
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.listBox = new System.Windows.Forms.ListBox();
            this.btnAdd = new System.Windows.Forms.Button();
            this.btnRemove = new System.Windows.Forms.Button();
            this.txtAdd = new System.Windows.Forms.TextBox();
            this.btnRecognise = new System.Windows.Forms.Button();
            this.btnLearn = new System.Windows.Forms.Button();
            this.Grid = new System.Windows.Forms.PictureBox();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.lblErr = new System.Windows.Forms.Label();
            this.txtErr = new System.Windows.Forms.TextBox();
            this.txtRounds = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.txtLearn = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.txtHiddens = new System.Windows.Forms.TextBox();
            this.btn_load = new System.Windows.Forms.Button();
            this.btn_save = new System.Windows.Forms.Button();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            ((System.ComponentModel.ISupportInitialize)(this.Grid)).BeginInit();
            this.groupBox1.SuspendLayout();
            this.SuspendLayout();
            // 
            // listBox
            // 
            this.listBox.FormattingEnabled = true;
            this.listBox.Location = new System.Drawing.Point(12, 12);
            this.listBox.Name = "listBox";
            this.listBox.Size = new System.Drawing.Size(175, 342);
            this.listBox.TabIndex = 1;
            this.listBox.DoubleClick += new System.EventHandler(this.listBox_DoubleClick);
            // 
            // btnAdd
            // 
            this.btnAdd.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnAdd.Location = new System.Drawing.Point(12, 386);
            this.btnAdd.Name = "btnAdd";
            this.btnAdd.Size = new System.Drawing.Size(82, 27);
            this.btnAdd.TabIndex = 2;
            this.btnAdd.Text = "+";
            this.btnAdd.UseVisualStyleBackColor = true;
            this.btnAdd.Click += new System.EventHandler(this.btnAdd_Click);
            // 
            // btnRemove
            // 
            this.btnRemove.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnRemove.Location = new System.Drawing.Point(105, 386);
            this.btnRemove.Name = "btnRemove";
            this.btnRemove.Size = new System.Drawing.Size(82, 27);
            this.btnRemove.TabIndex = 3;
            this.btnRemove.Text = "-";
            this.btnRemove.UseVisualStyleBackColor = true;
            this.btnRemove.Click += new System.EventHandler(this.btnRemove_Click);
            // 
            // txtAdd
            // 
            this.txtAdd.Location = new System.Drawing.Point(12, 360);
            this.txtAdd.Name = "txtAdd";
            this.txtAdd.Size = new System.Drawing.Size(175, 20);
            this.txtAdd.TabIndex = 4;
            // 
            // btnRecognise
            // 
            this.btnRecognise.Location = new System.Drawing.Point(12, 560);
            this.btnRecognise.Name = "btnRecognise";
            this.btnRecognise.Size = new System.Drawing.Size(175, 23);
            this.btnRecognise.TabIndex = 5;
            this.btnRecognise.Text = "Recognise";
            this.btnRecognise.UseVisualStyleBackColor = true;
            this.btnRecognise.Click += new System.EventHandler(this.btnRecognise_Click);
            // 
            // btnLearn
            // 
            this.btnLearn.Location = new System.Drawing.Point(12, 589);
            this.btnLearn.Name = "btnLearn";
            this.btnLearn.Size = new System.Drawing.Size(175, 23);
            this.btnLearn.TabIndex = 6;
            this.btnLearn.Text = "Learn";
            this.btnLearn.UseVisualStyleBackColor = true;
            this.btnLearn.Click += new System.EventHandler(this.btnLearn_Click);
            // 
            // Grid
            // 
            this.Grid.BackColor = System.Drawing.SystemColors.ControlDark;
            this.Grid.Location = new System.Drawing.Point(212, 12);
            this.Grid.Name = "Grid";
            this.Grid.Size = new System.Drawing.Size(500, 600);
            this.Grid.TabIndex = 8;
            this.Grid.TabStop = false;
            this.Grid.Paint += new System.Windows.Forms.PaintEventHandler(this.Grid_Paint);
            this.Grid.MouseDown += new System.Windows.Forms.MouseEventHandler(this.Grid_MouseDown);
            this.Grid.MouseMove += new System.Windows.Forms.MouseEventHandler(this.Grid_MouseMove);
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.lblErr);
            this.groupBox1.Controls.Add(this.txtErr);
            this.groupBox1.Controls.Add(this.txtRounds);
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Controls.Add(this.txtLearn);
            this.groupBox1.Controls.Add(this.label1);
            this.groupBox1.Controls.Add(this.txtHiddens);
            this.groupBox1.Location = new System.Drawing.Point(12, 467);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(175, 87);
            this.groupBox1.TabIndex = 9;
            this.groupBox1.TabStop = false;
            // 
            // lblErr
            // 
            this.lblErr.AutoSize = true;
            this.lblErr.Location = new System.Drawing.Point(6, 62);
            this.lblErr.Name = "lblErr";
            this.lblErr.Size = new System.Drawing.Size(50, 13);
            this.lblErr.TabIndex = 6;
            this.lblErr.Text = "Err. Limit:";
            // 
            // txtErr
            // 
            this.txtErr.Location = new System.Drawing.Point(56, 59);
            this.txtErr.Name = "txtErr";
            this.txtErr.Size = new System.Drawing.Size(29, 20);
            this.txtErr.TabIndex = 5;
            this.txtErr.Text = "0.05";
            // 
            // txtRounds
            // 
            this.txtRounds.Location = new System.Drawing.Point(102, 32);
            this.txtRounds.Name = "txtRounds";
            this.txtRounds.Size = new System.Drawing.Size(67, 20);
            this.txtRounds.TabIndex = 4;
            this.txtRounds.Text = "20000";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(6, 39);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(37, 13);
            this.label2.TabIndex = 3;
            this.label2.Text = "Learn:";
            // 
            // txtLearn
            // 
            this.txtLearn.Location = new System.Drawing.Point(56, 36);
            this.txtLearn.Name = "txtLearn";
            this.txtLearn.Size = new System.Drawing.Size(29, 20);
            this.txtLearn.TabIndex = 2;
            this.txtLearn.Text = "0.5";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(6, 16);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(44, 13);
            this.label1.TabIndex = 1;
            this.label1.Text = "Hidden:";
            // 
            // txtHiddens
            // 
            this.txtHiddens.Location = new System.Drawing.Point(56, 13);
            this.txtHiddens.Name = "txtHiddens";
            this.txtHiddens.Size = new System.Drawing.Size(29, 20);
            this.txtHiddens.TabIndex = 0;
            this.txtHiddens.Text = "10";
            // 
            // btn_load
            // 
            this.btn_load.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btn_load.Location = new System.Drawing.Point(12, 419);
            this.btn_load.Name = "btn_load";
            this.btn_load.Size = new System.Drawing.Size(82, 27);
            this.btn_load.TabIndex = 10;
            this.btn_load.Text = "Load";
            this.btn_load.UseVisualStyleBackColor = true;
            this.btn_load.Click += new System.EventHandler(this.btn_load_Click);
            // 
            // btn_save
            // 
            this.btn_save.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btn_save.Location = new System.Drawing.Point(105, 419);
            this.btn_save.Name = "btn_save";
            this.btn_save.Size = new System.Drawing.Size(82, 27);
            this.btn_save.TabIndex = 11;
            this.btn_save.Text = "Save";
            this.btn_save.UseVisualStyleBackColor = true;
            this.btn_save.Click += new System.EventHandler(this.btn_save_Click);
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.FileName = "openFileDialog1";
            this.openFileDialog1.RestoreDirectory = true;
            // 
            // PatternRecognition
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(724, 628);
            this.Controls.Add(this.btn_save);
            this.Controls.Add(this.btn_load);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.Grid);
            this.Controls.Add(this.btnLearn);
            this.Controls.Add(this.btnRecognise);
            this.Controls.Add(this.txtAdd);
            this.Controls.Add(this.btnRemove);
            this.Controls.Add(this.btnAdd);
            this.Controls.Add(this.listBox);
            this.DoubleBuffered = true;
            this.Name = "PatternRecognition";
            this.Text = "PatternRecognition";
            ((System.ComponentModel.ISupportInitialize)(this.Grid)).EndInit();
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        private System.Windows.Forms.ListBox listBox;
        private System.Windows.Forms.Button btnAdd;
        private System.Windows.Forms.Button btnRemove;
        private System.Windows.Forms.TextBox txtAdd;
        private System.Windows.Forms.Button btnRecognise;
        private System.Windows.Forms.Button btnLearn;
        private System.Windows.Forms.PictureBox Grid;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Label lblErr;
        private System.Windows.Forms.TextBox txtErr;
        private System.Windows.Forms.TextBox txtRounds;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox txtLearn;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox txtHiddens;
        private System.Windows.Forms.Button btn_load;
        private System.Windows.Forms.Button btn_save;
        private System.Windows.Forms.OpenFileDialog openFileDialog1;
    }
}

